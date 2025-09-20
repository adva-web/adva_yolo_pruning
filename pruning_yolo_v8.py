import logging
from ultralytics import YOLO
import torch

from yolov8_utils import build_mini_net, extract_conv_weights_norm, get_all_conv2d_layers, get_raw_objects_debug_v8, aggregate_activations_from_matches, prune_conv2d_layer_in_yolo, get_conv_bn_pairs, extract_bn_gamma
from yolo_layer_pruner import YoloLayerPruner
from clustering import select_optimal_components, kmedoids_fasterpam
import random
import numpy as np


logger = logging.getLogger("yolov8_pruning")
fh = logging.FileHandler('log.txt', mode='w')
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s:%(lineno)d - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.setLevel(logging.INFO)

# logger = logging.getLogger("yolov8_pruning")
# logging.basicConfig(level=logging.INFO)
def apply_pruning_v8_prune_lowest_activation(model_path, train_data, valid_data, classes, last_layer_idx=3, k_values=None):
    """
    Prune the channels with the lowest mean activation scores for each Conv2d layer.
    For each k in k_values, prune k channels with the lowest mean activation.
    """
    import numpy as np

    logger.info("Starting activation-based pruning for YOLOv8 model.")
    if k_values is None:
        k_values = list(range(4, 20, 4))  # Default: prune 4, 6, 8, ..., 18 channels

    for k in k_values:
        print(f"\n=== Pruning {k} channels with lowest mean activation per layer ===")
        model = YOLO(model_path)
        torch_model = model.model
        detection_model = torch_model.model  # nn.Sequential of blocks

        last_layer_idx = 2
        for layer_idx in range(2, last_layer_idx + 1):
            logger.info(f"Pruning block #{layer_idx}..")
            sliced_block = detection_model[:layer_idx]
            conv_layers = get_all_conv2d_layers(sliced_block)
            processed_convs = set()
            processed_convs.add(0)  
            for conv_idx, conv_layer in enumerate(conv_layers):
                num_channels = conv_layer.weight.shape[0]
                if num_channels <= k:
                    print(f"Skipping layer {conv_idx}: not enough channels to prune {k}.")
                    continue

                if conv_idx in processed_convs:
                    continue    
                processed_convs.add(conv_idx)

                # Build mini-net and get activations
                try:
                    mini_net = build_mini_net(sliced_block, conv_layer)
                except ValueError:
                    logger.error(f"Failed to build mini-network for conv2d layer #{conv_idx} of block {layer_idx - 1}. Skipping this layer.")
                    continue

                train_matched_objs, _ = get_raw_objects_debug_v8(model, mini_net, train_data)
                train_activations = aggregate_activations_from_matches(train_matched_objs, classes)

                # Save activation distributions
                with open("channel_activation_distributions_layer2.txt", "a") as f:
                    f.write(f"Layer {conv_idx} activation distributions:\n")
                    for ch in range(conv_layer.weight.shape[0]):
                        f.write(f"  Channel {ch}:\n")
                        for cls in range(len(classes)):
                            acts = train_activations.get(ch, {}).get(cls, [])
                            f.write(f"    Class {cls}: {acts}\n")

                # Compute mean activation per channel
                # train_activations: Dict[channel][class] -> List[float]
                mean_activations = []
                for ch in range(num_channels):
                    all_acts = []
                    for acts in train_activations.get(ch, {}).values():
                        all_acts.extend(acts)
                    if all_acts:
                        mean_activations.append(np.mean(all_acts))
                    else:
                        mean_activations.append(0.0)  # If no activations, treat as zero

                mean_activations = np.array(mean_activations)
                indices_sorted = np.argsort(mean_activations)  # ascending order
                indices_to_keep = sorted(indices_sorted[k:])  # keep all except k lowest
                indices_to_prune = indices_sorted[:k]
                
                # Save pruned channels
                with open("pruned_channels_per_step_layer3.txt", "a") as f:
                    f.write(f"Layer in activ {conv_idx} pruned channels this step: {list(indices_to_prune)}\n")

                # Prune
                model = prune_conv2d_layer_in_yolo(model, conv_idx, indices_to_keep)
                print(f"Pruned layer {conv_idx} (kept {len(indices_to_keep)} channels)")

                # Optionally retrain after each layer
                model.train(data="pruning/data/VOC_adva.yaml", epochs=3, verbose=False)
                print(f"Retrained model for 3 epochs after pruning layer {conv_idx}")

        # Final evaluation
        final_metrics = model.val(data="pruning/data/VOC_adva.yaml", verbose=False)
        logger.info(f"Final metrics after pruning: {final_metrics.results_dict}")
        print("DEBUG: Final evaluation complete.")

        # Save results
        with open("pruning_log_lowest_activation_layer3.txt", "a") as f:
            f.write(f"k={k}, mAP_0.5:0.95={final_metrics.results_dict.get('metrics/mAP50-95(B)', None)}, "
                    f"mAP_0.5={final_metrics.results_dict.get('metrics/mAP50(B)', None)}, "
                    f"precision={final_metrics.results_dict.get('metrics/precision(B)', None)}, "
                    f"recall={final_metrics.results_dict.get('metrics/recall(B)', None)}\n")

    return model

def apply_pruning_v8_fix_k_medoids(model_path, train_data, valid_data, classes, last_layer_idx=3):
    logger.info("Starting the pruning process for YOLOv8 model.")
    k_values = list(range(20, 32, 2))  
    for k in k_values:
        print(f"\n=== Pruning with k={k} components ===")
        # Load model
        model = YOLO(model_path)
        torch_model = model.model
        logger.debug("Model loaded successfully.")
        detection_model = torch_model.model  # YOLOv8: model.model.model is the nn.Sequential of blocks
        logger.debug("Accessed the detection model from YOLOv8.")
        
        last_layer_idx = 2
        for layer_idx in range(2, last_layer_idx + 1):
            logger.info(f"Starting to prune block #{layer_idx}..")
            sliced_block = detection_model[:layer_idx]
            print(f"DEBUG: Sliced block {layer_idx} with {len(sliced_block)} layers.")
            #New part 
            all_conv_layers = get_all_conv2d_layers(model)
            conv_layers = get_all_conv2d_layers(sliced_block)

            processed_convs = set()
            processed_convs.add(0)
            for conv_idx, conv_layer in enumerate(conv_layers):
                print("conv_idx:", conv_idx)
                print("conv_layer:", conv_layer)
                if conv_idx in processed_convs:
                        continue

                processed_convs.add(conv_idx)
                print("processed_convs:", processed_convs)

                logger.info(f"Starting to prune conv2d layer #{conv_idx} of block {layer_idx - 1}..")
                with open("pruning_fix_k_medoids.txt", "a") as f:
                    num_channels_before = conv_layer.weight.shape[0]
                    f.write(f"Layer {conv_idx} channels before pruning: {num_channels_before}\n")

                try:
                    mini_net = build_mini_net(sliced_block, conv_layer)
                    print(f"DEBUG: Built mini_net for block {layer_idx}, layer {conv_idx}")
                except ValueError:
                    logger.error(f"Failed to build mini-network for conv2d layer #{conv_idx} of block {layer_idx - 1}. Skipping this layer.")
                    print(f"DEBUG: Failed to build mini_net for block {layer_idx}, layer {conv_idx}")
                    continue

                layer_weights = extract_conv_weights_norm(conv_layer)
                print(f"DEBUG: Extracted weights norm for layer {conv_idx}, shape: {layer_weights.shape}")

                train_matched_objs, train_unmatched_objs = get_raw_objects_debug_v8(model, mini_net, train_data)
                print(f"DEBUG: Got raw objects for layer {conv_idx}")

                train_activations = aggregate_activations_from_matches(train_matched_objs, classes)
                print(f"DEBUG: Aggregated activations for layer {layer_idx}, shape: {train_activations.shape if hasattr(train_activations, 'shape') else type(train_activations)}")


                if not train_activations or all(len(v) == 0 for v in train_activations.values()):
                    logger.warning(f"No matched activations for layer {layer_idx}, skipping pruning for this layer.")
                    continue

                graph_space = YoloLayerPruner(activations=train_activations).create_layer_space()
                print(f"DEBUG: Created graph space for layer {layer_idx}")

                print("layer_weights.shape:", layer_weights.shape)
                print("graph_space['reduced_matrix'].shape:", graph_space['reduced_matrix'].shape)
                print("len(train_activations):", len(train_activations))

                # Cluster and select k components
                k_medoids = kmedoids_fasterpam(graph_space['reduced_matrix'], k)
                indices_to_keep = k_medoids['medoids'].tolist()
                
                model = prune_conv2d_layer_in_yolo(model, conv_idx, indices_to_keep)
                all_conv_layers = get_all_conv2d_layers(model)

                print(f"DEBUG: Pruned layer {conv_idx}")

                pruned_layer = all_conv_layers[conv_idx]

                num_channels_after = (pruned_layer.weight.abs().sum(dim=(1,2,3)) != 0).sum().item()
                with open("pruning_fix_k_medoids.txt", "a") as f:
                    f.write(f"Layer {conv_idx} channels after pruning: {num_channels_after}\n")
                    f.write(f"Pruned {num_channels_before - num_channels_after} channels in layer {conv_idx}\n")
                    f.write("=== all_conv_layers ===")
                    for idx, layer in enumerate(all_conv_layers):
                        f.write(f"all_conv_layers[{idx}]: id={id(layer)}, shape={tuple(layer.weight.shape)}")

                model.train(data="pruning/data/VOC_adva.yaml", epochs=5, verbose=False)
                print(f"DEBUG: Retrained model for 5 epoch after pruning layer {conv_idx}")

        # Final evaluation
        final_metrics = model.val(data="pruning/data/VOC_adva.yaml", verbose=False)
        logger.info(f"Final metrics after pruning: {final_metrics.results_dict}")
        print("DEBUG: Final evaluation complete.")

        with open("pruning_fix_k_medoids.txt", "a") as f:
            f.write(f"k={k}, mAP_0.5:0.95={final_metrics.results_dict.get('metrics/mAP50-95(B)', None)}, "
            f"mAP_0.5={final_metrics.results_dict.get('metrics/mAP50(B)', None)}, "
            f"precision={final_metrics.results_dict.get('metrics/precision(B)', None)}, "
            f"recall={final_metrics.results_dict.get('metrics/recall(B)', None)}\n")

    return model

def apply_pruning_v8_fix_number(model_path, train_data, valid_data, classes, last_layer_idx=3):
    logger.info("Starting the pruning process for YOLOv8 model.")
    select_components_based_on_mss = False

    k_values = list(range(4, 20, 2))  # Try pruning 4, 6, 8, ..., 18 channels
    for k in k_values:
        print(f"\n=== Pruning with k={k} channels per layer ===")
        # Load model
        model = YOLO(model_path)
        torch_model = model.model
        logger.debug("Model loaded successfully.")
        detection_model = torch_model.model  # YOLOv8: model.model.model is the nn.Sequential of blocks
        logger.debug("Accessed the detection model from YOLOv8.") 

        last_layer_idx = 2
        for layer_idx in range(2, last_layer_idx + 1):
            logger.info(f"Starting to prune block #{layer_idx}..")
            sliced_block = detection_model[:layer_idx]
            #New part 
            all_conv_layers = get_all_conv2d_layers(model)
            conv_layers = get_all_conv2d_layers(sliced_block)

            print("=== all_conv_layers ===")
            for idx, layer in enumerate(all_conv_layers):
                print(f"all_conv_layers[{idx}]: id={id(layer)}, shape={tuple(layer.weight.shape)}")

            print("=== conv_layers ===")
            for idx, layer in enumerate(conv_layers):
                print(f"conv_layers[{idx}]: id={id(layer)}, shape={tuple(layer.weight.shape)}")

            processed_convs = set()
            processed_convs.add(0)

            for conv_idx, conv_layer in enumerate(conv_layers):
                print("conv_idx:", conv_idx)
                print("conv_layer:", conv_layer)

                if conv_idx in processed_convs:
                        continue
                processed_convs.add(conv_idx)

                logger.info(f"Starting to prune conv2d layer #{conv_idx} of block {layer_idx - 1}..")

                with open("pruning_log_fix_prunned_rand_3_epoch.txt", "a") as f:
                    num_channels_before = conv_layer.weight.shape[0]
                    f.write(f"Layer {conv_idx} channels before pruning: {num_channels_before}\n")
                if num_channels_before <= k:
                    print(f"Skipping layer {conv_idx}: not enough channels to prune {k}.")
                    continue

                indices_to_keep = sorted(random.sample(range(num_channels_before), num_channels_before - k))
                # indices_to_keep = list(range(num_channels_before - k))
                model = prune_conv2d_layer_in_yolo(model, conv_idx, indices_to_keep)
                all_conv_layers = get_all_conv2d_layers(model)

                print(f"DEBUG: Pruned layer {conv_idx}")

                pruned_layer = all_conv_layers[conv_idx]
                num_channels_after = (pruned_layer.weight.abs().sum(dim=(1,2,3)) != 0).sum().item()
                with open("pruning_log_fix_prunned_rand_3_epoch.txt", "a") as f:
                    f.write(f"Layer {conv_idx} channels after pruning: {num_channels_after}\n")
                    f.write(f"Pruned {num_channels_before - num_channels_after} channels in layer {conv_idx}\n")
                    f.write("=== all_conv_layers ===")
                    for idx, layer in enumerate(all_conv_layers):
                        f.write(f"all_conv_layers[{idx}]: id={id(layer)}, shape={tuple(layer.weight.shape)}")

                model.train(data="pruning/data/VOC_adva.yaml", epochs=3, verbose=False)
                print(f"DEBUG: Retrained model for 3 epoch after pruning layer {conv_idx}")
                
        # Final evaluation
        final_metrics = model.val(data="pruning/data/VOC_adva.yaml", verbose=False)
        logger.info(f"Final metrics after pruning: {final_metrics.results_dict}")
        print("DEBUG: Final evaluation complete.")

        with open("pruning_log_fix_prunned_rand_3_epoch.txt", "a") as f:
            f.write(f"k={k}, mAP_0.5:0.95={final_metrics.results_dict.get('metrics/mAP50-95(B)', None)}, "
            f"mAP_0.5={final_metrics.results_dict.get('metrics/mAP50(B)', None)}, "
            f"precision={final_metrics.results_dict.get('metrics/precision(B)', None)}, "
            f"recall={final_metrics.results_dict.get('metrics/recall(B)', None)}\n")

    return model

def apply_pruning_v8(model_path, train_data, valid_data, classes, last_layer_idx=3):
    logger.info("Starting the pruning process for YOLOv8 model.")
    k_default_value = 50
    select_components_based_on_mss = False
    # Load model
    model = YOLO(model_path)
    torch_model = model.model
    logger.debug("Model loaded successfully.")
    detection_model = torch_model.model  # YOLOv8: model.model.model is the nn.Sequential of blocks
    logger.debug("Accessed the detection model from YOLOv8.") 

    last_layer_idx = 5
    for layer_idx in range(5, last_layer_idx + 1):
        logger.info(f"Starting to prune block #{layer_idx}..")
        sliced_block = detection_model[:layer_idx]
        print(f"DEBUG: Sliced block {layer_idx} with {len(sliced_block)} layers.")
        #New part 
        all_conv_layers = get_all_conv2d_layers(model)
        conv_layers = get_all_conv2d_layers(sliced_block)

        print("=== all_conv_layers ===")
        for idx, layer in enumerate(all_conv_layers):
            print(f"all_conv_layers[{idx}]: id={id(layer)}, shape={tuple(layer.weight.shape)}")

        print("=== conv_layers ===")
        for idx, layer in enumerate(conv_layers):
            print(f"conv_layers[{idx}]: id={id(layer)}, shape={tuple(layer.weight.shape)}")

        processed_convs = set()
        processed_convs.add(0)

        print("processed_convs:", processed_convs)
        for conv_idx, conv_layer in enumerate(conv_layers):
            print("conv_idx:", conv_idx)
            print("conv_layer:", conv_layer)
            if conv_idx in processed_convs:
                    continue

            processed_convs.add(conv_idx)
            print("processed_convs:", processed_convs)

            logger.info(f"Starting to prune conv2d layer #{conv_idx} of block {layer_idx - 1}..")
            with open("pruning_layer6.txt", "a") as f:
                num_channels_before = conv_layer.weight.shape[0]
                f.write(f"Layer {conv_idx} channels before pruning: {num_channels_before}\n")

            try:
                mini_net = build_mini_net(sliced_block, conv_layer)
                print(f"DEBUG: Built mini_net for block {layer_idx}, layer {conv_idx}")
            except ValueError:
                logger.error(f"Failed to build mini-network for conv2d layer #{conv_idx} of block {layer_idx - 1}. Skipping this layer.")
                print(f"DEBUG: Failed to build mini_net for block {layer_idx}, layer {conv_idx}")
                continue

            layer_weights = extract_conv_weights_norm(conv_layer)
            print(f"DEBUG: Extracted weights norm for layer {conv_idx}, shape: {layer_weights.shape}")

            train_matched_objs, train_unmatched_objs = get_raw_objects_debug_v8(model, mini_net, train_data)
            logger.info(f"Total number of training objects used:\n"
                        f"Train matched objects: {len(train_matched_objs)} - "
                        f"Train un-matched objects: {len(train_unmatched_objs)} - "
                        f"Percentage of matched objects overall: [{len(train_matched_objs) / (len(train_matched_objs) + len(train_unmatched_objs))}].")
            print(f"DEBUG: Got raw objects for layer {conv_idx}")

            train_activations = aggregate_activations_from_matches(train_matched_objs, classes)
            print(f"DEBUG: Aggregated activations for layer {layer_idx}, shape: {train_activations.shape if hasattr(train_activations, 'shape') else type(train_activations)}")


            if not train_activations or all(len(v) == 0 for v in train_activations.values()):
                logger.warning(f"No matched activations for layer {layer_idx}, skipping pruning for this layer.")
                continue

            graph_space = YoloLayerPruner(activations=train_activations).create_layer_space()
            print(f"DEBUG: Created graph space for layer {layer_idx}")

            print("layer_weights.shape:", layer_weights.shape)
            print("graph_space['reduced_matrix'].shape:", graph_space['reduced_matrix'].shape)
            print("len(train_activations):", len(train_activations))

            if not select_components_based_on_mss:
            # TODO: In the original implementation, k_default_value was not being used, so this if-else block did the same thing. Need to ask what the purpose was or check in the original algorithm.
                optimal_components = select_optimal_components(graph_space, layer_weights, len(train_activations), k_default_value)
                select_components_based_on_mss = True
            else:
                optimal_components = select_optimal_components(graph_space, layer_weights, len(train_activations), -1)
            
            logger.info(f"Number of optimal components for conv2d layer #{conv_idx}: {len(optimal_components)}")

            print(f"DEBUG: Selected {len(optimal_components)} optimal components for layer {conv_idx}")
            print("Main model id:", id(model))
            model = prune_conv2d_layer_in_yolo(model, conv_idx, optimal_components)
            all_conv_layers = get_all_conv2d_layers(model)

            print(f"DEBUG: Pruned layer {conv_idx}")

            pruned_layer = all_conv_layers[conv_idx]
            num_channels_after = (pruned_layer.weight.abs().sum(dim=(1,2,3)) != 0).sum().item()
            with open("pruning_layer6.txt", "a") as f:
                f.write(f"Layer {conv_idx} channels after pruning: {num_channels_after}\n")
                f.write(f"Pruned {num_channels_before - num_channels_after} channels in layer {conv_idx}\n")

            #Evaluate after pruning this layer
            pruned_metrics = model.val(data="pruning/data/VOC_adva.yaml", verbose=False)
            logger.info(f"After pruning layer  & **before** re-train conv layer {conv_idx}: {pruned_metrics.results_dict}")
            print(f"DEBUG: Evaluated pruned model after layer {conv_idx}")

            model.train(data="pruning/data/VOC_adva.yaml", epochs=20, verbose=False)
            print(f"DEBUG: Retrained model for 1 epoch after pruning layer {conv_idx}")

    # Final evaluation
    final_metrics = model.val(data="pruning/data/VOC_adva.yaml", verbose=False)
    logger.info(f"Final metrics after pruning: {final_metrics.results_dict}")
    print("DEBUG: Final evaluation complete.")
    with open("pruning_layer6.txt", "a") as f:
        f.write(f"Final mAP_0.5:0.95={final_metrics.results_dict.get('metrics/mAP50-95(B)', None)}, "
                f"mAP_0.5={final_metrics.results_dict.get('metrics/mAP50(B)', None)}, "
                f"precision={final_metrics.results_dict.get('metrics/precision(B)', None)}, "
                f"recall={final_metrics.results_dict.get('metrics/recall(B)', None)}\n")
        # Per-class mAP@0.5
        if hasattr(final_metrics, "maps"):
            f.write("Per-class mAP@0.5:\n")
            for idx, class_map in enumerate(final_metrics.maps):
                f.write(f"Class {idx}: {class_map:.4f}\n")
        else:
            f.write("Per-class mAP not available in results.\n")

    return model

def apply_pruning_v8_activation_based(model_path, train_data, valid_data, classes,
                                      block_idx=6, conv_idx_within_block=0,
                                      k_default_value=50, select_components_based_on_mss=False):
    """
    Prune a specific Conv2d layer in a YOLOv8 block using activation-based pruning
    with MSS or top-k strategy. Zeroes out pruned channels using user-defined pruning function.
    """
    model = YOLO(model_path)
    torch_model = model.model
    detection_model = torch_model.model  # nn.Sequential of blocks

    logger.info(f"Starting activation-based pruning for block #{block_idx}, conv #{conv_idx_within_block}")

    block = detection_model[block_idx]
    all_conv_layers = get_all_conv2d_layers(model)
    conv_layers_in_block = get_all_conv2d_layers(block)

    if conv_idx_within_block >= len(conv_layers_in_block):
        logger.warning(f"Block {block_idx} has only {len(conv_layers_in_block)} conv layers; got conv_idx {conv_idx_within_block}")
        return

    conv_layer = conv_layers_in_block[conv_idx_within_block]

    try:
        sliced_block = detection_model[:block_idx + 1]
        mini_net = build_mini_net(sliced_block, conv_layer)
        print(f"DEBUG: Built mini_net for block {block_idx}, layer {conv_idx_within_block}")
    except ValueError:
        logger.error(f"Failed to build mini-network for conv layer #{conv_idx_within_block} in block {block_idx}")
        return

    layer_weights = extract_conv_weights_norm(conv_layer)
    train_matched_objs, train_unmatched_objs = get_raw_objects_debug_v8(model, mini_net, train_data)
    train_activations = aggregate_activations_from_matches(train_matched_objs, classes)

    if not train_activations or all(len(v) == 0 for v in train_activations.values()):
        logger.warning(f"No matched activations for conv layer {conv_idx_within_block} in block {block_idx}. Skipping.")
        return

    graph_space = YoloLayerPruner(activations=train_activations).create_layer_space()

    # Select channels to keep
    if not select_components_based_on_mss:
        optimal_components = select_optimal_components(
            graph_space, layer_weights, len(train_activations), k_default_value
        )
        logger.info(f"Selected top-{k_default_value} channels by activation")
    else:
        optimal_components = select_optimal_components(
            graph_space, layer_weights, len(train_activations), -1
        )
        logger.info(f"Selected optimal components based on MSS")

    # Map local conv to global conv index
    try:
        global_conv_idx = next(i for i, conv in enumerate(all_conv_layers) if conv is conv_layer)
    except StopIteration:
        logger.error("Could not find global conv index for pruning.")
        return

    model = prune_conv2d_layer_in_yolo(model, global_conv_idx, optimal_components)
    print(f"Pruned conv #{global_conv_idx} (kept {len(optimal_components)} channels)")
    #Evaluate after pruning this layer
    pruned_metrics = model.val(data="pruning/data/VOC_adva.yaml", verbose=False)
    logger.info(f"After pruning layer  & **before** re-train conv layer {global_conv_idx}: {pruned_metrics.results_dict}")
    print(f"DEBUG: Evaluated pruned model after layer {global_conv_idx}")

    model.train(data="pruning/data/VOC_adva.yaml", epochs=5, verbose=False)
    print(f"DEBUG: Retrained model for 5 epoch after pruning layer {global_conv_idx}")

    # Final evaluation
    final_metrics = model.val(data="pruning/data/VOC_adva.yaml", verbose=False)
    logger.info(f"Final metrics after pruning: {final_metrics.results_dict}")
    print("DEBUG: Final evaluation complete.")
    with open("pruning_layer4.txt", "a") as f:
        f.write(f"Final mAP_0.5:0.95={final_metrics.results_dict.get('metrics/mAP50-95(B)', None)}, "
                f"mAP_0.5={final_metrics.results_dict.get('metrics/mAP50(B)', None)}, "
                f"precision={final_metrics.results_dict.get('metrics/precision(B)', None)}, "
                f"recall={final_metrics.results_dict.get('metrics/recall(B)', None)}\n")
        # Per-class mAP@0.5
        if hasattr(final_metrics, "maps"):
            f.write("Per-class mAP@0.5:\n")
            for idx, class_map in enumerate(final_metrics.maps):
                f.write(f"Class {idx}: {class_map:.4f}\n")
        else:
            f.write("Per-class mAP not available in results.\n")

    return model

def apply_bn_pruning_v8(model_path, train_data, valid_data, classes, last_layer_idx=11):
    logger.info("Starting the BN+Conv pruning process for YOLOv8 model.")
    k_default_value = 50

    # Load model
    model = YOLO(model_path)
    torch_model = model.model
    logger.debug("Model loaded successfully.")
    detection_model = torch_model.model  # YOLOv8: model.model.model is the nn.Sequential of blocks
    logger.debug("Accessed the detection model from YOLOv8.")
    last_layer_idx = 2
    select_components_based_on_mss = False


    for layer_idx in range(2, last_layer_idx + 1):
        logger.info(f"Starting to prune block #{layer_idx}..")
        sliced_block = detection_model[:layer_idx]
        conv_bn_pairs = get_conv_bn_pairs(sliced_block)

        processed_convs = set()
        processed_convs.add(0)

        for conv_idx, (conv_layer, bn_layer) in enumerate(conv_bn_pairs):
            if conv_idx in processed_convs:
                continue
            processed_convs.add(conv_idx)
            logger.info(f"Starting to prune conv2d+bn layer #{conv_idx} of block {layer_idx - 1}..")
            
            with open("pruning_log_v7.txt", "a") as f:
                num_channels_before = conv_layer.weight.shape[0]
                f.write(f"Layer {conv_idx} channels before pruning: {num_channels_before}\n")
            try:
                mini_net = build_mini_net(sliced_block, conv_layer)
                print(f"DEBUG: Built mini_net for block {layer_idx}, layer {conv_idx}")
            except ValueError:
                logger.error(f"Failed to build mini-network for conv2d layer #{conv_idx} of block {layer_idx - 1}. Skipping this layer.")
                print(f"DEBUG: Failed to build mini_net for block {layer_idx}, layer {conv_idx}")
                continue

            # Use BN gamma for pruning instead of conv weights
            layer_gammas = extract_bn_gamma(bn_layer)
            layer_weights = extract_conv_weights_norm(conv_layer)
            print(f"DEBUG: Extracted BN gamma for layer {conv_idx}, shape: {layer_gammas.shape}")
            with open("pruning_log_v7.txt", "a") as f:
                f.write(f"Layer {conv_idx} BN gamma shape: {layer_gammas.shape}\n")
                f.write(f"Layer {conv_idx} BN gamma values: {layer_gammas.tolist()}\n")
                f.write(f"Layer {conv_idx} weights values: {layer_weights.tolist()}\n")


            train_matched_objs, train_unmatched_objs = get_raw_objects_debug_v8(model, mini_net, train_data)
            logger.info(f"Total number of training objects used:\n"
                        f"Train matched objects: {len(train_matched_objs)} - "
                        f"Train un-matched objects: {len(train_unmatched_objs)} - "
                        f"Percentage of matched objects overall: [{len(train_matched_objs) / (len(train_matched_objs) + len(train_unmatched_objs))}].")
            print(f"DEBUG: Got raw objects for layer {conv_idx}")

            train_activations = aggregate_activations_from_matches(train_matched_objs, classes)
            print(f"DEBUG: Aggregated activations for layer {conv_idx}, shape: {train_activations.shape if hasattr(train_activations, 'shape') else type(train_activations)}")

            graph_space = YoloLayerPruner(activations=train_activations).create_layer_space()
            print(f"DEBUG: Created graph space for layer {conv_idx}")

            if not select_components_based_on_mss:
                optimal_components = select_optimal_components(graph_space, layer_gammas, len(train_activations), k_default_value)
                select_components_based_on_mss = True
            else:
                optimal_components = select_optimal_components(graph_space, layer_gammas, len(train_activations), -1)

            logger.info(f"Number of optimal components for conv2d+bn layer #{conv_idx}: {len(optimal_components)}")
            print(f"DEBUG: Selected {len(optimal_components)} optimal components for layer {conv_idx}")

            model = prune_conv2d_layer_in_yolo(model, conv_idx, optimal_components)
            print(f"DEBUG: Pruned layer {conv_idx}")

            # # Evaluate after pruning this layer
            # pruned_metrics = model.val(data="pruning/data/VOC_adva.yaml", verbose=False)
            # logger.info(f"After pruning layer  & **before** re-train conv+bn layer {conv_idx}: {pruned_metrics.results_dict}")
            # print(f"DEBUG: Evaluated pruned model after layer {conv_idx}")

            num_channels_after = (conv_layer.weight.abs().sum(dim=(1,2,3)) != 0).sum().item()
            with open("pruning_log_v6.txt", "a") as f:
                f.write(f"Layer {conv_idx} channels after pruning: {num_channels_after}\n")
                f.write(f"Pruned {num_channels_before - num_channels_after} channels in layer {conv_idx}\n")

            # Optionally retrain for 1 epoch
            model.train(data="pruning/data/VOC_adva.yaml", epochs=5, verbose=False)
            print(f"DEBUG: Retrained model for 5 epoch after pruning layer {conv_idx}")

        # Final evaluation
    final_metrics = model.val(data="pruning/data/VOC_adva.yaml", verbose=False)
    logger.info(f"Final metrics after pruning: {final_metrics.results_dict}")
    print("DEBUG: Final evaluation complete.")

    return model

def apply_pruning_v8_prune_lowest_gamma(model_path, train_data, valid_data, classes, last_layer_idx=3, k_values=None):
    """
    Prune the channels with the lowest BN gamma values for each Conv2d+BN pair.
    For each k in k_values, prune k channels with the lowest gamma.
    """

    logger.info("Starting BN-gamma-based pruning for YOLOv8 model.")
    if k_values is None:
        k_values = list(range(4, 20, 4))  # Default: prune 4, 6, 8, ..., 18 channels

    for k in k_values:
        print(f"\n=== Pruning {k} channels with lowest gamma per layer ===")
        model = YOLO(model_path)
        torch_model = model.model
        detection_model = torch_model.model  # nn.Sequential of blocks

        last_layer_idx = 2
        for layer_idx in range(2, last_layer_idx + 1):
            logger.info(f"Pruning block #{layer_idx}..")
            sliced_block = detection_model[:layer_idx]
            conv_bn_pairs = get_conv_bn_pairs(sliced_block)

            processed_convs = set()
            processed_convs.add(0)  
            for conv_idx, (conv_layer, bn_layer) in enumerate(conv_bn_pairs):
                num_channels = conv_layer.weight.shape[0]
                if num_channels <= k:
                    print(f"Skipping layer {conv_idx}: not enough channels to prune {k}.")
                    continue

                if conv_idx in processed_convs:
                    continue
                processed_convs.add(conv_idx)

                # Get BN gamma values
                gammas = extract_bn_gamma(bn_layer)
                with open("pruning_log_lowest_gamma.txt", "a") as f:
                    f.write(f"Layer {conv_idx} gammas before pruning: {gammas}\n")
                # Indices of channels to keep: those with the highest gamma
                indices_sorted = np.argsort(gammas)  # ascending order
                indices_to_keep = sorted(indices_sorted[k:])  # keep all except k lowest
                indices_to_prune = indices_sorted[:k]

                with open("pruned_channels_per_step.txt", "a") as f:
                    f.write(f"Layer {conv_idx} in gamma pruned channels this step: {list(indices_to_prune)}\n")

                # Prune
                model = prune_conv2d_layer_in_yolo(model, conv_idx, indices_to_keep)
                print(f"Pruned layer {conv_idx} (kept {len(indices_to_keep)} channels)")

                # Optionally retrain after each layer
                model.train(data="pruning/data/VOC_adva.yaml", epochs=3, verbose=False)
                print(f"Retrained model for 3 epochs after pruning layer {conv_idx}")

        # Final evaluation
        final_metrics = model.val(data="pruning/data/VOC_adva.yaml", verbose=False)
        logger.info(f"Final metrics after pruning: {final_metrics.results_dict}")
        print("DEBUG: Final evaluation complete.")

        # Save results
        with open("pruning_log_lowest_gamma.txt", "a") as f:
            f.write(f"k={k}, mAP_0.5:0.95={final_metrics.results_dict.get('metrics/mAP50-95(B)', None)}, "
                    f"mAP_0.5={final_metrics.results_dict.get('metrics/mAP50(B)', None)}, "
                    f"precision={final_metrics.results_dict.get('metrics/precision(B)', None)}, "
                    f"recall={final_metrics.results_dict.get('metrics/recall(B)', None)}\n")

    return model

def apply_gamma_pruning_on_block_zeroed(model_path, block_idx, k_values=None):
    """
    For each k in k_values, prune (zero out) the k channels with the lowest gamma
    in all Conv2d+BN pairs inside a specific block.
    Uses the user-defined prune_conv2d_layer_in_yolo (non-structural).
    """
    if k_values is None:
        k_values = list(range(128, 132, 4))  

    for k in k_values:
        print(f"\n===== Pruning k={k} in block #{block_idx} using gamma values =====")
        model = YOLO(model_path)
        torch_model = model.model
        detection_model = torch_model.model

        # Get all Conv2d layers for global indexing
        all_conv_layers = get_all_conv2d_layers(detection_model)

        # Get the block to prune
        block = detection_model[block_idx]
        conv_bn_pairs = get_conv_bn_pairs(block)

        for pair_local_idx, (conv_layer, bn_layer) in enumerate(conv_bn_pairs):
            num_channels = conv_layer.weight.shape[0]
            if num_channels <= k:
                print(f"Skipping local conv #{pair_local_idx} in block {block_idx}: not enough channels.")
                continue

            gammas = extract_bn_gamma(bn_layer)
            indices_sorted = np.argsort(gammas)
            indices_to_keep = sorted(indices_sorted[k:])

            # Find the conv_layer's index in the global list
            try:
                global_conv_idx = next(i for i, conv in enumerate(all_conv_layers) if conv is conv_layer)
            except StopIteration:
                print(f"Conv layer not found in global list. Skipping.")
                continue

            model = prune_conv2d_layer_in_yolo(model, global_conv_idx, indices_to_keep)
            # Optionally retrain after each layer
            model.train(data="pruning/data/VOC_adva.yaml", epochs=20, verbose=False)

        # Final evaluation
        final_metrics = model.val(data="pruning/data/VOC_adva.yaml", verbose=False)
        logger.info(f"Final metrics after pruning: {final_metrics.results_dict}")
        print("DEBUG: Final evaluation complete.")

        # Save results
        with open("pruning_log_lowest_gamma.txt", "a") as f:
            f.write(f"k={k}, mAP_0.5:0.95={final_metrics.results_dict.get('metrics/mAP50-95(B)', None)}, "
                    f"mAP_0.5={final_metrics.results_dict.get('metrics/mAP50(B)', None)}, "
                    f"precision={final_metrics.results_dict.get('metrics/precision(B)', None)}, "
                    f"recall={final_metrics.results_dict.get('metrics/recall(B)', None)}\n")

    return model

def apply_gamma_pruning_iter(
    model_path,
    block_idx=5,
    conv_in_block_idx=0,
    finetune_epochs=5
):
    """
    Prune 10% of channels (by lowest gamma, computed once) from a specific Conv2d layer in a block,
    until 50% of the original channels are pruned. Finetune after each pruning step.
    """
    import torch.nn as nn
    import numpy as np

    logger.info(f"Starting iterative gamma pruning for block {block_idx}, Conv2d #{conv_in_block_idx}.")
    model = YOLO(model_path)
    torch_model = model.model
    detection_model = torch_model.model  # nn.Sequential of blocks

    block = detection_model[block_idx]
    conv_layers_in_block = get_all_conv2d_layers(block)
    if conv_in_block_idx >= len(conv_layers_in_block):
        logger.warning(f"conv_in_block_idx {conv_in_block_idx} out of range for block {block_idx}.")
        return model
    target_conv_layer = conv_layers_in_block[conv_in_block_idx]

    # Find the index of the target Conv2d in the full model
    all_conv_layers = get_all_conv2d_layers(model)
    target_conv_idx = all_conv_layers.index(target_conv_layer)

    # Get original number of channels
    original_num_channels = target_conv_layer.weight.shape[0]
    num_to_prune_total = original_num_channels // 2  # 50%
    num_to_prune_per_iter = max(1, original_num_channels // 10)  # 10%

    # --- Compute gamma and pruning order ONCE ---
    # Find the BatchNorm after this Conv2d in the block
    bn_layer = None
    found = False
    for sublayer in block.children():
        if found and isinstance(sublayer, nn.BatchNorm2d):
            bn_layer = sublayer
            break
        if sublayer is target_conv_layer:
            found = True
    if bn_layer is None:
        logger.warning("No BatchNorm2d found after target Conv2d. Cannot prune by gamma.")
        return model

    gamma = bn_layer.weight.detach().cpu().numpy()
    indices_sorted = np.argsort(gamma)  # ascending order: lowest gamma first

    # --- Iteratively prune 10% each time, following the fixed gamma order ---
    pruned_so_far = 0
    while pruned_so_far < num_to_prune_total:
        # Always refresh the reference to the current pruned layer
        all_conv_layers = get_all_conv2d_layers(model)
        target_conv_layer = all_conv_layers[target_conv_idx]
        current_num_channels = target_conv_layer.weight.shape[0]
        print(f"Current number of channels in pruned layer: {current_num_channels}")
        
        prune_count = min(num_to_prune_per_iter, num_to_prune_total - pruned_so_far)
        indices_to_prune = indices_sorted[pruned_so_far:pruned_so_far + prune_count]
        indices_to_keep = sorted(set(range(original_num_channels)) - set(indices_sorted[:pruned_so_far + prune_count]))

        logger.info(f"Pruning {prune_count} channels (total pruned: {pruned_so_far + prune_count}/{num_to_prune_total})")
        print(f"Pruning indices: {indices_to_prune}")
        model = prune_conv2d_layer_in_yolo(model, target_conv_idx, indices_to_keep)
        pruned_so_far += prune_count

        # Check which channels are all zero
        all_conv_layers = get_all_conv2d_layers(model)
        conv_layer = all_conv_layers[target_conv_idx]  # or your target index
        weight_sums = conv_layer.weight.data.abs().sum(dim=(1,2,3))
        zeroed_channels = (weight_sums == 0).nonzero(as_tuple=True)[0].tolist()
        print(f"Zeroed output channels in Conv2d layer {target_conv_idx}: {zeroed_channels}")


        # Finetune after each pruning step
        model.train(data="pruning/data/VOC_adva.yaml", epochs=finetune_epochs, verbose=False)
        print(f"Finetuned for {finetune_epochs} epochs after pruning.")
                
        iter_metrics = model.val(data="pruning/data/VOC_adva.yaml", verbose=False)
        logger.info(f"Finetuned metrics after pruning: {iter_metrics.results_dict}")

    # Final evaluation
    final_metrics = model.val(data="pruning/data/VOC_adva.yaml", verbose=False)
    logger.info(f"Final metrics after iterative pruning: {final_metrics.results_dict}")
    print("DEBUG: Final evaluation complete.")

    return model

def prune_conv2d_in_block_with_activations(
    model_path,
    train_data,
    valid_data,
    classes,
    block_idx=5,             # Index of the block in detection_model
    conv_in_block_idx=0,     # Index of the Conv2d layer within the block
    log_file="pruning_block_conv.txt"):
    """
    Prune a specific Conv2d layer inside a block, aligning with activation extraction.
    """
    import torch.nn as nn

    logger.info(f"Pruning Conv2d in block {block_idx}, Conv2d #{conv_in_block_idx}.")

    model = YOLO(model_path)
    torch_model = model.model
    detection_model = torch_model.model  # nn.Sequential of blocks
    select_components_based_on_mss = False
    # 1. Get the target block and its Conv2d layers
    block = detection_model[block_idx]
    conv_layers_in_block = get_all_conv2d_layers(block)
    if conv_in_block_idx >= len(conv_layers_in_block):
        logger.warning(f"conv_in_block_idx {conv_in_block_idx} out of range for block {block_idx}.")
        return model
    target_conv_layer = conv_layers_in_block[conv_in_block_idx]

    # 2. Build sliced_block: all blocks before, plus partial block up to target Conv2d
    blocks_up_to = list(detection_model[:block_idx])
    submodules = []
    conv_count = 0
    for sublayer in block.children():
        submodules.append(sublayer)
        if isinstance(sublayer, nn.Conv2d):
            if conv_count == conv_in_block_idx:
                break
            conv_count += 1
    partial_block = nn.Sequential(*submodules)
    sliced_block = nn.Sequential(*(blocks_up_to + [partial_block]))

    # 3. Find the index of the target Conv2d in the full model
    all_conv_layers = get_all_conv2d_layers(model)
    target_conv_idx = all_conv_layers.index(target_conv_layer)

    # 4. Extract activations using sliced_block
    try:
        mini_net = build_mini_net(sliced_block, target_conv_layer)
        print(f"DEBUG: Built mini_net for block {block_idx}, conv {conv_in_block_idx}")
    except ValueError:
        logger.error(f"Failed to build mini-network for conv2d layer #{conv_in_block_idx} of block {block_idx}. Skipping this layer.")
        return model

    layer_weights = extract_conv_weights_norm(target_conv_layer)
    train_matched_objs, train_unmatched_objs = get_raw_objects_debug_v8(model, mini_net, train_data)
    train_activations = aggregate_activations_from_matches(train_matched_objs, classes)

    if not train_activations or all(len(v) == 0 for v in train_activations.values()):
        logger.warning(f"No matched activations for block {block_idx}, conv {conv_in_block_idx}, skipping pruning for this layer.")
        return model
    
    graph_space = YoloLayerPruner(activations=train_activations).create_layer_space()

    if not select_components_based_on_mss:
        optimal_components = select_optimal_components(graph_space, layer_weights, len(train_activations), -1)
        select_components_based_on_mss = True
    else:
        optimal_components = select_optimal_components(graph_space, layer_weights, len(train_activations), -1)
            
    # 5. Prune the Conv2d layer using the provided indices
    print(f"Pruning Conv2d layer {target_conv_idx} (block {block_idx}, conv {conv_in_block_idx}) with indices: {optimal_components}")
    model = prune_conv2d_layer_in_yolo(model, target_conv_idx, optimal_components)
    all_conv_layers = get_all_conv2d_layers(model)
    pruned_layer = all_conv_layers[target_conv_idx]
    num_channels_after = (pruned_layer.weight.abs().sum(dim=(1,2,3)) != 0).sum().item()

    with open(log_file, "a") as f:
        f.write(f"Layer {target_conv_idx} channels after pruning: {num_channels_after}\n")
        f.write(f"Pruned {pruned_layer.weight.shape[0] - num_channels_after} channels in layer {target_conv_idx}\n")

    # 6. Evaluate after pruning
    pruned_metrics = model.val(data="pruning/data/VOC_adva.yaml", verbose=False)
    logger.info(f"After pruning layer & **before** re-train conv layer {target_conv_idx}: {pruned_metrics.results_dict}")
    print(f"DEBUG: Evaluated pruned model after layer {target_conv_idx}")

    model.train(data="pruning/data/VOC_adva.yaml", epochs=20, verbose=False)
    print(f"DEBUG: Retrained model for 5 epochs after pruning layer {target_conv_idx}")

    # Final evaluation
    final_metrics = model.val(data="pruning/data/VOC_adva.yaml", verbose=False)
    logger.info(f"Final metrics after pruning: {final_metrics.results_dict}")
    print("DEBUG: Final evaluation complete.")
    with open(log_file, "a") as f:
        f.write(f"Final mAP_0.5:0.95={final_metrics.results_dict.get('metrics/mAP50-95(B)', None)}, "
                f"mAP_0.5={final_metrics.results_dict.get('metrics/mAP50(B)', None)}, "
                f"precision={final_metrics.results_dict.get('metrics/precision(B)', None)}, "
                f"recall={final_metrics.results_dict.get('metrics/recall(B)', None)}\n")
        if hasattr(final_metrics, "maps"):
            f.write("Per-class mAP@0.5:\n")
            for idx, class_map in enumerate(final_metrics.maps):
                f.write(f"Class {idx}: {class_map:.4f}\n")
        else:
            f.write("Per-class mAP not available in results.\n")
            
    return model

def apply_50_percent_gamma_pruning_blocks_3_4(model_path, layers_to_prune=3):
    """
    Prune 50% of channels (based on lowest gamma values) from 2-3 layers in blocks 3-4.
    
    Args:
        model_path: Path to the YOLO model
        layers_to_prune: Number of layers to prune (2 or 3)
    
    Returns:
        Pruned and retrained model
    """
    if layers_to_prune < 3 or layers_to_prune > 4:
        raise ValueError("layers_to_prune must be 3 or 4")
    
    print(f"\n===== Pruning 50% of channels from {layers_to_prune} layers in blocks 3-4 =====")
    
    # Load model
    model = YOLO(model_path)
    torch_model = model.model
    detection_model = torch_model.model
    
    # Get all Conv2d layers for global indexing
    all_conv_layers = get_all_conv2d_layers(detection_model)
    
    # Collect all conv-bn pairs from blocks 3 and 4 with original model indexing
    target_blocks = [3, 4, 5]
    all_available_pairs = []
    
    # Create a mapping of conv layers to their original indices for reference
    original_conv_layer_mapping = {}
    for original_idx, conv_layer in enumerate(all_conv_layers):
        original_conv_layer_mapping[id(conv_layer)] = original_idx
    
    print(f"Original model has {len(all_conv_layers)} Conv2d layers total")
    
    for block_idx in target_blocks:
        if block_idx >= len(detection_model):
            print(f"Warning: Block index {block_idx} is out of range. Skipping.")
            continue
            
        block = detection_model[block_idx]
        conv_bn_pairs = get_conv_bn_pairs(block)
        
        print(f"\nAnalyzing Block {block_idx}:")
        print(f"  Found {len(conv_bn_pairs)} Conv2d+BN pairs in this block")
        
        for pair_local_idx, (conv_layer, bn_layer) in enumerate(conv_bn_pairs):
            num_channels = conv_layer.weight.shape[0]
            
            # Find original model index for this conv layer
            original_conv_idx = original_conv_layer_mapping.get(id(conv_layer), "Unknown")
            
            print(f"    Pair #{pair_local_idx}: {num_channels} channels, Original model index: {original_conv_idx}")
            
            # Skip layers with too few channels (need at least 4 channels to prune 50%)
            if num_channels < 4:
                print(f"    → Skipping: only {num_channels} channels (need ≥4 for 50% pruning)")
                continue
            
            all_available_pairs.append({
                'block_idx': block_idx,
                'pair_local_idx': pair_local_idx,
                'conv_layer': conv_layer,
                'bn_layer': bn_layer,
                'num_channels': num_channels,
                'original_model_idx': original_conv_idx
            })
    
    print(f"Found {len(all_available_pairs)} suitable Conv2d+BN pairs in blocks 3-4")
    
    if len(all_available_pairs) < layers_to_prune:
        print(f"Warning: Only {len(all_available_pairs)} pairs available, adjusting to prune all available layers")
        layers_to_prune = len(all_available_pairs)
    
    # Calculate gamma statistics for layer selection
    pairs_with_gamma_stats = []
    
    for pair_info in all_available_pairs:
        bn_layer = pair_info['bn_layer']
        gammas = extract_bn_gamma(bn_layer)
        avg_gamma = np.mean(gammas)
        
        pair_info['gammas'] = gammas
        pair_info['avg_gamma'] = avg_gamma
        pairs_with_gamma_stats.append(pair_info)
    
    # Sort by average gamma (lowest first) and select layers to prune
    pairs_with_gamma_stats.sort(key=lambda x: x['avg_gamma'])
    selected_pairs = pairs_with_gamma_stats[:layers_to_prune]
    
    print(f"\nSelected {len(selected_pairs)} layers for 50% pruning:")
    for i, pair_info in enumerate(selected_pairs):
        channels_to_remove = pair_info['num_channels'] // 2
        channels_to_keep = pair_info['num_channels'] - channels_to_remove
        print(f"  Layer {i+1}: Block {pair_info['block_idx']}, Local pair #{pair_info['pair_local_idx']}")
        print(f"    Original model index: {pair_info['original_model_idx']}")
        print(f"    Channels: {pair_info['num_channels']} → {channels_to_keep} (removing {channels_to_remove})")
        print(f"    Avg gamma: {pair_info['avg_gamma']:.6f}")
    
    # Apply 50% pruning to selected layers
    pruned_layers_details = []
    
    print(f"\n--- Starting Pruning Process ---")
    for idx, pair_info in enumerate(selected_pairs):
        conv_layer = pair_info['conv_layer']
        gammas = pair_info['gammas']
        num_channels = pair_info['num_channels']
        
        # Calculate how many channels to remove (50%)
        channels_to_remove = num_channels // 2
        channels_to_keep_count = num_channels - channels_to_remove
        
        # Find indices to keep (remove channels with lowest gamma values)
        indices_sorted = np.argsort(gammas)  # Sort by gamma value (lowest first)
        indices_to_keep = sorted(indices_sorted[channels_to_remove:])  # Keep the higher gamma channels
        
        # Find the conv_layer's index in the global list
        try:
            global_conv_idx = next(i for i, conv in enumerate(all_conv_layers) if conv is conv_layer)
        except StopIteration:
            print(f"Conv layer not found in global list. Skipping block {pair_info['block_idx']}, pair #{pair_info['pair_local_idx']}.")
            continue
        
        # Detailed pruning information
        print(f"\nPruning Layer {idx + 1}/{len(selected_pairs)}:")
        print(f"  - Block: {pair_info['block_idx']}")
        print(f"  - Local pair index: {pair_info['pair_local_idx']}")
        print(f"  - Original model index: {pair_info['original_model_idx']}")
        print(f"  - Current global conv layer index: {global_conv_idx}")
        print(f"  - Original channels: {num_channels}")
        print(f"  - Channels to remove: {channels_to_remove}")
        print(f"  - Channels to keep: {len(indices_to_keep)}")
        print(f"  - Avg gamma value: {pair_info['avg_gamma']:.6f}")
        print(f"  - Gamma range: {np.min(gammas):.6f} to {np.max(gammas):.6f}")
        
        # Store details for final summary
        pruned_layers_details.append({
            'block_idx': pair_info['block_idx'],
            'local_pair_idx': pair_info['pair_local_idx'],
            'original_model_idx': pair_info['original_model_idx'],
            'global_conv_idx': global_conv_idx,
            'original_channels': num_channels,
            'pruned_channels': channels_to_remove,
            'remaining_channels': len(indices_to_keep),
            'avg_gamma': pair_info['avg_gamma']
        })
        
        # Apply pruning
        model = prune_conv2d_layer_in_yolo(model, global_conv_idx, indices_to_keep)
        print(f"  ✓ Pruning applied successfully!")
        
        # Update all_conv_layers reference since model structure changed
        torch_model = model.model
        detection_model = torch_model.model
        all_conv_layers = get_all_conv2d_layers(detection_model)
    
    # Retrain after pruning all selected layers
    print(f"\nStarting retraining after 50% pruning of {len(selected_pairs)} layers...")
    model.train(data="pruning/data/VOC_adva.yaml", epochs=20, verbose=False)
    
    # Final evaluation
    print("Starting final evaluation...")
    final_metrics = model.val(data="pruning/data/VOC_adva.yaml", verbose=False)
    
    # Calculate total parameters pruned
    total_channels_before = sum(pair['num_channels'] for pair in selected_pairs)
    total_channels_after = sum(pair['num_channels'] // 2 for pair in selected_pairs)
    pruning_ratio = (total_channels_before - total_channels_after) / total_channels_before * 100
    
    print(f"\nDetailed Pruning Summary:")
    print(f"{'='*80}")
    print(f"{'Layer':<8} {'Block':<6} {'Local#':<7} {'Original#':<10} {'Current#':<9} {'Channels':<15} {'Gamma':<10}")
    print(f"{'-'*80}")
    for i, details in enumerate(pruned_layers_details):
        channels_info = f"{details['original_channels']}→{details['remaining_channels']}"
        print(f"{i+1:<8} {details['block_idx']:<6} {details['local_pair_idx']:<7} "
              f"{details['original_model_idx']:<10} {details['global_conv_idx']:<9} "
              f"{channels_info:<15} {details['avg_gamma']:<10.6f}")
    
    print(f"{'-'*80}")
    print(f"Overall Statistics:")
    print(f"  Layers pruned: {len(selected_pairs)}")
    print(f"  Total channels before: {total_channels_before}")
    print(f"  Total channels after: {total_channels_after}")
    print(f"  Overall pruning ratio: {pruning_ratio:.1f}%")
    print(f"{'='*80}")
    
    logger.info(f"Final metrics after 50% pruning: {final_metrics.results_dict}")
    print("DEBUG: Final evaluation complete.")
    
    # Enhanced log file with detailed information
    with open("pruning_log_50_percent_blocks_3_4.txt", "a") as f:
        f.write(f"\n--- Pruning Session ---\n")
        f.write(f"Layers pruned: {len(selected_pairs)}\n")
        f.write(f"Layer Details:\n")
        for i, details in enumerate(pruned_layers_details):
            f.write(f"  Layer {i+1}: Block {details['block_idx']}, Local #{details['local_pair_idx']}, "
                   f"Original model #{details['original_model_idx']}, Current #{details['global_conv_idx']}: "
                   f"{details['original_channels']}→{details['remaining_channels']} channels "
                   f"(gamma: {details['avg_gamma']:.6f})\n")
        f.write(f"Total channels: {total_channels_before}→{total_channels_after} ({pruning_ratio:.1f}% reduction)\n")
        f.write(f"Performance: mAP_0.5:0.95={final_metrics.results_dict.get('metrics/mAP50-95(B)', None)}, "
                f"mAP_0.5={final_metrics.results_dict.get('metrics/mAP50(B)', None)}, "
                f"precision={final_metrics.results_dict.get('metrics/precision(B)', None)}, "
                f"recall={final_metrics.results_dict.get('metrics/recall(B)', None)}\n")
        f.write(f"--- End Session ---\n\n")
    
    return model

def apply_activation_pruning_blocks_3_4(model_path, train_data, valid_data, classes, layers_to_prune=4):
    """
    Prune multiple layers in blocks 3-4 using activation-based pruning similar to apply_50_percent_gamma_pruning_blocks_3_4
    but using the prune_conv2d_in_block_with_activations strategy.
    
    Args:
        model_path: Path to the YOLO model
        train_data: Training data for activation extraction
        valid_data: Validation data 
        classes: List of class names
        layers_to_prune: Number of layers to prune (default 3)
    
    Returns:
        Pruned and retrained model
    """
    if layers_to_prune < 2 or layers_to_prune > 6:
        raise ValueError("layers_to_prune must be between 2 and 6")
    
    print(f"\n===== Activation-based pruning of {layers_to_prune} layers in blocks 3-4 =====")
    
    # Load model
    model = YOLO(model_path)
    torch_model = model.model
    detection_model = torch_model.model
    
    # Get all Conv2d layers for global indexing
    all_conv_layers = get_all_conv2d_layers(detection_model)
    
    # Collect all conv layers from target blocks with original model indexing
    target_blocks = [3, 4, 5 , 6 ]
    all_available_convs = []
    
    # Create a mapping of conv layers to their original indices for reference
    original_conv_layer_mapping = {}
    for original_idx, conv_layer in enumerate(all_conv_layers):
        original_conv_layer_mapping[id(conv_layer)] = original_idx
    
    print(f"Original model has {len(all_conv_layers)} Conv2d layers total")
    
    for block_idx in target_blocks:
        if block_idx >= len(detection_model):
            print(f"Warning: Block index {block_idx} is out of range. Skipping.")
            continue
            
        block = detection_model[block_idx]
        conv_layers_in_block = get_all_conv2d_layers(block)
        
        print(f"\nAnalyzing Block {block_idx}:")
        print(f"  Found {len(conv_layers_in_block)} Conv2d layers in this block")
        
        for conv_in_block_idx, conv_layer in enumerate(conv_layers_in_block):
            num_channels = conv_layer.weight.shape[0]
            
            # Find original model index for this conv layer
            original_conv_idx = original_conv_layer_mapping.get(id(conv_layer), "Unknown")
            
            print(f"    Conv #{conv_in_block_idx}: {num_channels} channels, Original model index: {original_conv_idx}")
            
            # Skip layers with too few channels (need at least 4 channels for meaningful pruning)
            if num_channels < 8:
                print(f"    → Skipping: only {num_channels} channels (need ≥8 for activation pruning)")
                continue
            
            all_available_convs.append({
                'block_idx': block_idx,
                'conv_in_block_idx': conv_in_block_idx,
                'conv_layer': conv_layer,
                'num_channels': num_channels,
                'original_model_idx': original_conv_idx
            })
    
    print(f"Found {len(all_available_convs)} suitable Conv2d layers in blocks 3-4")
    
    if len(all_available_convs) < layers_to_prune:
        print(f"Warning: Only {len(all_available_convs)} layers available, adjusting to prune all available layers")
        layers_to_prune = len(all_available_convs)
    
    # Select layers with most channels for activation-based pruning (often more impactful)
    all_available_convs.sort(key=lambda x: x['num_channels'], reverse=True)
    selected_convs = all_available_convs[:layers_to_prune]
    
    print(f"\nSelected {len(selected_convs)} layers for activation-based pruning:")
    for i, conv_info in enumerate(selected_convs):
        print(f"  Layer {i+1}: Block {conv_info['block_idx']}, Conv #{conv_info['conv_in_block_idx']}")
        print(f"    Original model index: {conv_info['original_model_idx']}")
        print(f"    Channels: {conv_info['num_channels']}")
    
    # Apply activation-based pruning to selected layers
    pruned_layers_details = []
    
    print(f"\n--- Starting Activation-Based Pruning Process ---")
    for idx, conv_info in enumerate(selected_convs):
        print(f"\nPruning Layer {idx + 1}/{len(selected_convs)}:")
        print(f"  - Block: {conv_info['block_idx']}")
        print(f"  - Conv in block index: {conv_info['conv_in_block_idx']}")
        print(f"  - Original model index: {conv_info['original_model_idx']}")
        print(f"  - Original channels: {conv_info['num_channels']}")
        
        # Store original model state to file temporarily for the function call
        temp_model_path = f"temp_model_state_{idx}.pt"
        model.save(temp_model_path)
        
        # Apply the activation-based pruning for this specific layer
        try:
            # Use the prune_conv2d_in_block_with_activations function
            model = prune_conv2d_in_block_with_activations(
                model_path=temp_model_path,
                train_data=train_data,
                valid_data=valid_data,
                classes=classes,
                block_idx=conv_info['block_idx'],
                conv_in_block_idx=conv_info['conv_in_block_idx'],
                log_file=f"pruning_activation_blocks_3_4_layer_{idx+1}.txt"
            )
            
            print(f"  ✓ Activation-based pruning applied successfully!")
            
            # Get updated channel count
            torch_model = model.model
            detection_model = torch_model.model
            all_conv_layers_updated = get_all_conv2d_layers(detection_model)
            
            # Find the pruned layer in the updated model
            if conv_info['original_model_idx'] < len(all_conv_layers_updated):
                pruned_layer = all_conv_layers_updated[conv_info['original_model_idx']]
                remaining_channels = (pruned_layer.weight.abs().sum(dim=(1,2,3)) != 0).sum().item()
            else:
                remaining_channels = "Unknown"
            
            # Store details for final summary
            pruned_layers_details.append({
                'block_idx': conv_info['block_idx'],
                'conv_in_block_idx': conv_info['conv_in_block_idx'],
                'original_model_idx': conv_info['original_model_idx'],
                'original_channels': conv_info['num_channels'],
                'remaining_channels': remaining_channels,
                'pruned_channels': conv_info['num_channels'] - remaining_channels if isinstance(remaining_channels, int) else "Unknown"
            })
            
        except Exception as e:
            print(f"  ✗ Error during activation-based pruning: {e}")
            logger.error(f"Failed to prune block {conv_info['block_idx']}, conv {conv_info['conv_in_block_idx']}: {e}")
            continue
        finally:
            # Clean up temporary file
            import os
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
    
    # Final evaluation
    print("Starting final evaluation...")
    final_metrics = model.val(data="pruning/data/VOC_adva.yaml", verbose=False)
    
    # Calculate total parameters pruned
    total_channels_before = sum(detail['original_channels'] for detail in pruned_layers_details)
    total_channels_after = sum(detail['remaining_channels'] for detail in pruned_layers_details if isinstance(detail['remaining_channels'], int))
    pruning_ratio = (total_channels_before - total_channels_after) / total_channels_before * 100 if total_channels_before > 0 else 0
    
    print(f"\nDetailed Activation-Based Pruning Summary:")
    print(f"{'='*80}")
    print(f"{'Layer':<8} {'Block':<6} {'Conv#':<7} {'Original#':<10} {'Channels':<15}")
    print(f"{'-'*80}")
    for i, details in enumerate(pruned_layers_details):
        channels_info = f"{details['original_channels']}→{details['remaining_channels']}"
        print(f"{i+1:<8} {details['block_idx']:<6} {details['conv_in_block_idx']:<7} "
              f"{details['original_model_idx']:<10} {channels_info:<15}")
    
    print(f"{'-'*80}")
    print(f"Overall Statistics:")
    print(f"  Layers pruned: {len(pruned_layers_details)}")
    print(f"  Total channels before: {total_channels_before}")
    print(f"  Total channels after: {total_channels_after}")
    print(f"  Overall pruning ratio: {pruning_ratio:.1f}%")
    print(f"{'='*80}")
    
    logger.info(f"Final metrics after activation-based pruning: {final_metrics.results_dict}")
    print("DEBUG: Final evaluation complete.")
    
    # Enhanced log file with detailed information
    with open("pruning_log_activation_blocks_3_4.txt", "a") as f:
        f.write(f"\n--- Activation-Based Pruning Session ---\n")
        f.write(f"Layers pruned: {len(pruned_layers_details)}\n")
        f.write(f"Layer Details:\n")
        for i, details in enumerate(pruned_layers_details):
            f.write(f"  Layer {i+1}: Block {details['block_idx']}, Conv #{details['conv_in_block_idx']}, "
                   f"Original model #{details['original_model_idx']}: "
                   f"{details['original_channels']}→{details['remaining_channels']} channels\n")
        f.write(f"Total channels: {total_channels_before}→{total_channels_after} ({pruning_ratio:.1f}% reduction)\n")
        f.write(f"Performance: mAP_0.5:0.95={final_metrics.results_dict.get('metrics/mAP50-95(B)', None)}, "
                f"mAP_0.5={final_metrics.results_dict.get('metrics/mAP50(B)', None)}, "
                f"precision={final_metrics.results_dict.get('metrics/precision(B)', None)}, "
                f"recall={final_metrics.results_dict.get('metrics/recall(B)', None)}\n")
        f.write(f"--- End Session ---\n\n")
    
    return model
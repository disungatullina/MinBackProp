import numpy as np
import torch
import time
import pymagsac
from tqdm import tqdm
from model_cl import *
from utils import *
from datasets import Dataset


def test(model, test_loader, opt):
    with torch.no_grad():
        avg_model_time = 0  # runtime of the network forward pass
        avg_ransac_time = 0  # runtime of RANSAC
        # essential matrix evaluation
        pose_losses = []
        avg_F1 = 0
        avg_inliers = 0
        epi_errors = []
        invalid_pairs = 0

        for idx, test_data in enumerate(tqdm(test_loader)):
            correspondences, K1, K2 = (
                test_data["correspondences"].to(opt.device),
                test_data["K1"].to(opt.device),
                test_data["K2"].to(opt.device),
            )
            im_size1, im_size2 = test_data["im_size1"].to(opt.device), test_data[
                "im_size2"
            ].to(opt.device)
            gt_E, gt_R, gt_t = (
                test_data["gt_E"].numpy(),
                test_data["gt_R"].numpy(),
                test_data["gt_t"].numpy(),
            )
            batch_size = correspondences.size(0)

            # predicted inlier probabilities and normalization.
            inlier_weights, _ = model(
                correspondences.float(),
                K1,
                K2,
                im_size1,
                im_size2,
                opt.prob,
                predict=False,
            )

            K1, K2 = K1.cpu().detach().numpy(), K2.cpu().detach().numpy()
            im_size1, im_size2 = (
                im_size1.cpu().detach().numpy(),
                im_size2.cpu().detach().numpy(),
            )
            # sorted_indices_batch = torch.argsort(logits, descending=True, dim=1).cpu().detach()
            ransac_time = 0
            correspondences = correspondences.cpu().detach()
            for b in range(batch_size):
                inliers = torch.zeros(1, 2000, 1)  # inlier mask of the estimated model
                # sorted_indices = sorted_indices_batch[b]
                weights = inlier_weights[b].cpu().detach().numpy()
                sorted_indices = np.argsort(weights)[::-1]

                # === CASE ESSENTIAL MATRIX =========================================
                pts1 = correspondences[b, 0:2].squeeze().numpy().T
                pts2 = correspondences[b, 2:4].squeeze().numpy().T

                # rank the points according to their probabilities
                sorted_pts1 = pts1[sorted_indices]
                sorted_pts2 = pts2[sorted_indices]
                weights = weights[sorted_indices]

                start_time = time.time()
                E, mask, save_samples = pymagsac.findEssentialMatrix(
                    np.ascontiguousarray(sorted_pts1).astype(
                        np.float64
                    ),  # pts[sorted_indices]
                    np.ascontiguousarray(sorted_pts2).astype(np.float64),
                    K1[b],
                    K2[b],
                    float(im_size1[b][0]),
                    float(im_size1[b][1]),
                    float(im_size2[b][0]),
                    float(im_size2[b][1]),
                    # probabilities=get_probabilities(sorted_pts1.shape[0])
                    probabilities=weights,
                    use_magsac_plus_plus=True,
                    sigma_th=opt.threshold,
                    sampler_id=3,
                    save_samples=True,
                )
                ransac_time += time.time() - start_time
                # count inlier number
                incount = np.sum(mask)
                incount /= correspondences.size(2)

                if incount == 0:
                    E = np.identity(3)
                else:
                    # update inliers
                    # inliers[0, :, 0] = torch.tensor(mask)
                    sorted_index = sorted_indices[mask]
                    inliers[0, sorted_index, 0] = 1

                # pts for recovering the pose
                pts1 = correspondences[b, 0:2].numpy()
                pts2 = correspondences[b, 2:4].numpy()

                pts1_1 = pts1.transpose(2, 1, 0)
                pts2_2 = pts2.transpose(2, 1, 0)

                inliers = inliers.byte().numpy().ravel()
                K = np.eye(3)
                R = np.eye(3)
                t = np.zeros((3, 1))

                # evaluation of relative pose (essential matrix)
                cv2.recoverPose(
                    E,
                    np.ascontiguousarray(pts1_1).astype(np.float64),
                    np.ascontiguousarray(pts2_2).astype(np.float64),
                    K,
                    R,
                    t,
                    inliers,
                )

                dR, dT = pose_error(R, gt_R[b], t, gt_t[b])
                pose_losses.append(max(float(dR), float(dT)))

            avg_ransac_time += ransac_time / batch_size

        print(
            "\nAvg. Model Time: %dms"
            % (avg_model_time / len(test_loader) * 1000 + 0.00000001)
        )
        print(
            "Avg. RANSAC Time: %dms"
            % (avg_ransac_time / len(test_loader) * 1000 + 0.00000001)
        )

        # calculate AUC of pose losses
        thresholds = [5, 10, 20]
        AUC_scores = AUC(
            losses=pose_losses, thresholds=thresholds, binsize=5
        )  # opt.evalbinsize)
        print("\n=== Relative Pose Accuracy ===========================")
        print(
            "AUC for %ddeg/%ddeg/%ddeg: %.2f/%.2f/%.2f\n"
            % (
                thresholds[0],
                thresholds[1],
                thresholds[2],
                AUC_scores[0],
                AUC_scores[1],
                AUC_scores[2],
            )
        )

        # write evaluation results to fil
        if not os.path.isdir("results/" + opt.model):
            os.makedirs("results/" + opt.model)
        with open("results/" + opt.model + "/test.txt", "a", 1) as f:
            f.write(
                "%f %f %f %f ms "
                % (
                    AUC_scores[0],
                    AUC_scores[1],
                    AUC_scores[2],
                    avg_ransac_time / len(test_loader) * 1000,
                )
            )
            f.write("\n")


if __name__ == "__main__":
    # Parse the parameters
    parser = create_parser(description="Generalized Differentiable RANSAC.")
    opt = parser.parse_args()
    opt.device = torch.device(
        "cuda:0" if torch.cuda.is_available() and opt.device != "cpu" else "cpu"
    )
    print(f"Running on {opt.device}")

    # collect dataset list to be used for testing
    if opt.batch_mode:
        scenes = test_datasets
        print(
            "\n=== BATCH MODE: Doing evaluation on",
            len(scenes),
            "datasets. =================",
        )
    else:
        scenes = [opt.datasets]

    model = DeepRansac_CLNet(opt).to(opt.device)

    for seq in scenes:
        print(f"Working on {seq} with scoring {opt.scoring}")
        scene_data_path = os.path.join(opt.data_path)
        dataset = Dataset(
            [scene_data_path + "/" + seq + "/test_data_rs/"],
            opt.snn,
            nfeatures=opt.nfeatures,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            num_workers=0,
            pin_memory=False,
            shuffle=False,
        )
        print(f"Loading test data: {len(dataset)} image pairs.")

        # if opt.model is not None:
        model.load_state_dict(torch.load(opt.model, map_location=opt.device))
        model.eval()
        test(model, test_loader, opt)

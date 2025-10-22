Dual-Head Task + User CNN (with Anthropometric Normalization)
=============================================================

This Github repository contains a baseline Python script that demos the methodology implemented for this project.  
It performs joint VR task and user classification using a 1D CNN. It also includes anthropometric normalization
(arm-length and height scaling) before standard z-score normalization. A pre-trained model is uploaded to the repository
as well for a quick demo of the performance.

Main file:
    vr_dualhead_task_user_cnn_anthro.py

-------------------------------------------------------------
1.  Training Data Folder layout
-------------------------------------------------------------
root/
  user_id/
    trial_id/
      file.csv

root folder for training is "Data_train"

Each CSV should contain features such as imu_pos_*, imu_rot_*, rh_pos_*, rh_rot_*,
rh_*_curl, etc.  Left-hand (lh_*) data is ignored automatically.
Task names are inferred from filenames (e.g., Hammer_3.csv -> Hammer).

-------------------------------------------------------------
2.  Testing Data Folder layout
-------------------------------------------------------------
root/
  user_id/
    trial_id/
      file.csv

root folder for testing is "Data_test"

Note: "Data_train" and "Data_test" is mutually exclusive for strict data split.

Each CSV should contain features such as imu_pos_*, imu_rot_*, rh_pos_*, rh_rot_*,
rh_*_curl, etc.  Left-hand (lh_*) data is ignored automatically.
Task names are inferred from filenames (e.g., Hammer_3.csv -> Hammer).

-------------------------------------------------------------
3.  Installation
-------------------------------------------------------------
1)  Create or activate a Python environment.
2)  Install all dependencies:
        pip install -r requirements.txt


-------------------------------------------------------------
4.  Training example (Command Prompt)
-------------------------------------------------------------
No need to re-train model if testing with Example model provided.

If user would like to train new model:

    python vr_dualhead_task_user_cnn_anthro.py train ^
      --root "C:\path\to\your\Data_train" ^ 
      --model-out ".\models\dual_model_cnn_artifact.pkl" ^
      --epochs 30 ^
      --batch-size 128 ^
      --precision-target 0.98 ^
      --train-counts 6,2 ^
      --test-counts 2 ^
      --max-wins-per-file 200 ^
      --dropout 0.2 ^

  

Outputs:

      models\dual_model_cnn_artifact_cnn.keras
      models\dual_model_cnn_artifact.pkl
      models\exports\train_epoch_times.csv
      models\exports\train_summary.csv

- When train and test data are stored seperately, splits are not
  necessary when train/test, just use data path accordingly.
  User may split data in "Data_train" based on own preference.
  
-------------------------------------------------------------
5.  Testing example (Command Prompt)
-------------------------------------------------------------
Testing with Example model:

    python vr_dualhead_task_user_cnn_anthro.py test ^
      --root "C:\path\to\your\Data_test" ^
      --model-in ".\Example_model\dual_taskid_example_artifact.pkl" ^
      --require both ^
      --time-per-window

Testing with new-trained model:

    python vr_dualhead_task_user_cnn_anthro.py test ^
      --root "C:\path\to\your\Data_test" ^
      --model-in ".\models\dual_model_cnn_artifact.pkl" ^
      --require both ^
      --time-per-window


Useful split options when user decide to split data in "Data_train" for train, val, test. Change root acccordingly:

    --use-splits test ^
  
                   all          use all matching files
                   train        use train split only
                   val          use validation split only
                   test         use test split only
                   unused       trials not assigned to train/val/test
                   not-train    all non-train data (val + test + unused)


Other flags:
        time-per-window      flag to measure per-window inference latency
        export-dir <path>    chooses where all timing/latency CSVs are written   (Optional, default next to model-in if not set )


Outputs:
  test_window_metrics.csv
  test_per_trial_decisions.csv
  test_stream_latency_ms.csv
  test_decision_events.csv


-------------------------------------------------------------
6.  Normalization summary
-------------------------------------------------------------
1) Anthropometric normalization (per user):
   - Arm scale = 95th percentile of |rh_pos_(x,y,z)| from TRAIN windows.
   - Height scale = (95th - 5th) percentile range of imu_pos_z.
     If imu_pos_z missing, uses L2 range of imu_pos_(x,y,z).
   - Apply to all windows:
        rh_pos_* /= arm_scale
        imu_pos_* /= height_scale
   - Missing users or channels fall back to global medians.
   - Stored in artifact and reused during testing.

2) Channel z-score normalization:
   - Compute mean/std from TRAIN windows.
   - Apply to both train and test.
   - Small stds are clipped to 1.0 for stability.

Processing order:
   Anthropometric normalization  -->  Channel z-score normalization


-------------------------------------------------------------
7.  Example Model for Demo
-------------------------------------------------------------
A example model is uploaded for quick accessible demo purpose. A corresponding terminal log is also
included for quick reference.


-------------------------------------------------------------
8.  Output Exports
-------------------------------------------------------------
All results are automatically saved to the folder:
models\exports\

Key exported files include:
- train_summary.csv:  summary of epoch-level training and validation accuracy/loss
- train_epoch_times.csv:  per-epoch timing information (used to analyze training speed)
- test_window_metrics.csv:  accuracy, EER, precision, recall, and F1-score at window level
- test_per_trial_decisions.csv:  per-trial predicted user and task outcomes
- test_stream_latency_ms.csv:  latency statistics (p50, p90, p95, mean) for real-time performance
- test_decision_events.csv:  record of all per-window and per-trial classification events


All exported files are standard CSV format and can be opened in Excel or any data analysis tool.


-------------------------------------------------------------
9.  Notes
-------------------------------------------------------------
- If each user has exactly 10 trials and all are assigned to
  train (6) + val (2) + test (2), no "unused" trials remain.
  This is normal and expected.
- Use "--use-splits not-train" to test all non-train data.
- Warnings about ptxas.exe on Windows can be ignored.
- The built-in z-score normalization replaces the need for
  sklearn StandardScaler.

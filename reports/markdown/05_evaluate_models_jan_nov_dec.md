    'Evaluate Jan-Nov vs December Model Training Results.\n\nThis report:\n1. Loads training results from models/jan_nov_dec/training_results.json\n2. Recreates predictions for visualization\n3. Creates comprehensive visualizations:\n   - Line plots comparing all three models\n   - Scatter plots for each model\n   - Error distribution histograms\n   - Metrics comparison bar charts\n4. Exports metrics to metrics/jan_nov_dec_evaluation.json for DVC tracking\n\nThis script is designed to be run via generate_report.sh to produce HTML and Markdown outputs.\nUses Jupyter cell blocks (# %%) for interactive execution.\n'



    ======================================================================
    JAN-NOV vs DECEMBER MODEL EVALUATION REPORT
    ======================================================================
    Experiment: jan_nov_dec


    
    Loading results from: /home/chris/research/load-forecast/models/jan_nov_dec/training_results.json
    Training timestamp: 2025-12-22T20:48:27.894697


    
    ======================================================================
    EXPERIMENT CONFIGURATION
    ======================================================================
    Test year: 2023
    Test month: 12
    Random seed: 42


    
    ======================================================================
    LOADING DATA FOR VISUALIZATION
    ======================================================================


    Loaded data shape: (31705, 133)
    Full data shape (with NaN): (35041, 133)


    
    ======================================================================
    RECREATING PREDICTIONS FOR VISUALIZATION
    ======================================================================
    Recreated train/test split
      Train: 28732 rows
      Test: 2973 rows


    [0]	validation_0-rmse:1.41152	validation_1-rmse:1.31583


    [1]	validation_0-rmse:1.03185	validation_1-rmse:0.96623


    [2]	validation_0-rmse:0.77738	validation_1-rmse:0.73386


    [3]	validation_0-rmse:0.61092	validation_1-rmse:0.58601


    [4]	validation_0-rmse:0.50663	validation_1-rmse:0.49471


    [5]	validation_0-rmse:0.44259	validation_1-rmse:0.44318


    [6]	validation_0-rmse:0.40365	validation_1-rmse:0.41286


    [7]	validation_0-rmse:0.38039	validation_1-rmse:0.39891


    [8]	validation_0-rmse:0.36587	validation_1-rmse:0.38995


    [9]	validation_0-rmse:0.35617	validation_1-rmse:0.38468


    [10]	validation_0-rmse:0.34930	validation_1-rmse:0.38267


    [11]	validation_0-rmse:0.34300	validation_1-rmse:0.38077


    [12]	validation_0-rmse:0.33711	validation_1-rmse:0.38113


    [13]	validation_0-rmse:0.33299	validation_1-rmse:0.38075


    [14]	validation_0-rmse:0.32945	validation_1-rmse:0.38164


    [15]	validation_0-rmse:0.32433	validation_1-rmse:0.38221


    [16]	validation_0-rmse:0.32244	validation_1-rmse:0.38305


    [17]	validation_0-rmse:0.31801	validation_1-rmse:0.38397


    [18]	validation_0-rmse:0.31418	validation_1-rmse:0.38391


    [19]	validation_0-rmse:0.31131	validation_1-rmse:0.38588


    [20]	validation_0-rmse:0.30763	validation_1-rmse:0.38520


    [21]	validation_0-rmse:0.30486	validation_1-rmse:0.38523


    [22]	validation_0-rmse:0.30290	validation_1-rmse:0.38595


    [23]	validation_0-rmse:0.30104	validation_1-rmse:0.38730


    Predictions recreated successfully


    
    ======================================================================
    EXPORTING METRICS FOR DVC TRACKING
    ======================================================================
    ✓ Metrics saved to: metrics/jan_nov_dec_evaluation.json


    
    ======================================================================
    RESULTS SUMMARY
    ======================================================================
    
    Model                RMSE         MAE          R²          
    ------------------------------------------------------------
    Baseline             0.1815       0.1068       0.9520      
    Direct XGBoost       0.1892       0.1088       0.9479      
    OpenSTEF XGBoost     0.1928       0.1139       0.9459      



    
![png](05_evaluate_models_jan_nov_dec_files/05_evaluate_models_jan_nov_dec_10_0.png)
    



    
![png](05_evaluate_models_jan_nov_dec_files/05_evaluate_models_jan_nov_dec_11_0.png)
    



    
![png](05_evaluate_models_jan_nov_dec_files/05_evaluate_models_jan_nov_dec_12_0.png)
    



    
![png](05_evaluate_models_jan_nov_dec_files/05_evaluate_models_jan_nov_dec_13_0.png)
    


    
    ======================================================================
    EVALUATION REPORT COMPLETE
    ======================================================================
    
    ✓ Analyzed December 2023 test period
    ✓ Compared 3 models: Baseline, Direct XGBoost, OpenSTEF XGBoost
    ✓ Metrics exported to: metrics/jan_nov_dec_evaluation.json
    
    Key Findings:
      - Best RMSE: 0.1815 (Baseline Persistence)
      - Direct XGBoost RMSE improvement over baseline: -4.22%
      - OpenSTEF XGBoost RMSE improvement over baseline: -6.22%


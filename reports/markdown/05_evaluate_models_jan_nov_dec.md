    'Evaluate Jan-Nov vs December Model Training Results.\n\nThis report:\n1. Loads training results from models/jan_nov_dec/training_results.json\n2. Recreates predictions for visualization\n3. Creates comprehensive visualizations:\n   - Line plots comparing all three models\n   - Scatter plots for each model\n   - Error distribution histograms\n   - Metrics comparison bar charts\n4. Exports metrics to metrics/jan_nov_dec_evaluation.json for DVC tracking\n\nThis script is designed to be run via generate_report.sh to produce HTML and Markdown outputs.\nUses Jupyter cell blocks (# %%) for interactive execution.\n'



    ======================================================================
    JAN-NOV vs DECEMBER MODEL EVALUATION REPORT
    ======================================================================
    Experiment: jan_nov_dec


    
    Loading results from: /home/chris/research/load-forecast/models/jan_nov_dec/training_results.json
    Training timestamp: 2025-12-23T11:41:11.041076


    
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


    [0]	validation_0-rmse:1.39063	validation_1-rmse:1.44060


    [1]	validation_0-rmse:1.01642	validation_1-rmse:1.06249


    [2]	validation_0-rmse:0.76571	validation_1-rmse:0.81334


    [3]	validation_0-rmse:0.60221	validation_1-rmse:0.65435


    [4]	validation_0-rmse:0.49835	validation_1-rmse:0.55659


    [5]	validation_0-rmse:0.43552	validation_1-rmse:0.50038


    [6]	validation_0-rmse:0.39780	validation_1-rmse:0.47083


    [7]	validation_0-rmse:0.37443	validation_1-rmse:0.45347


    [8]	validation_0-rmse:0.36054	validation_1-rmse:0.44678


    [9]	validation_0-rmse:0.35021	validation_1-rmse:0.44298


    [10]	validation_0-rmse:0.34335	validation_1-rmse:0.44271


    [11]	validation_0-rmse:0.33704	validation_1-rmse:0.44103


    [12]	validation_0-rmse:0.33155	validation_1-rmse:0.44056


    [13]	validation_0-rmse:0.32747	validation_1-rmse:0.43987


    [14]	validation_0-rmse:0.32353	validation_1-rmse:0.43947


    [15]	validation_0-rmse:0.32038	validation_1-rmse:0.44002


    [16]	validation_0-rmse:0.31721	validation_1-rmse:0.44001


    [17]	validation_0-rmse:0.31338	validation_1-rmse:0.44147


    [18]	validation_0-rmse:0.31132	validation_1-rmse:0.44140


    [19]	validation_0-rmse:0.30716	validation_1-rmse:0.44267


    [20]	validation_0-rmse:0.30531	validation_1-rmse:0.44267


    [21]	validation_0-rmse:0.30245	validation_1-rmse:0.44308


    [22]	validation_0-rmse:0.29986	validation_1-rmse:0.44371


    [23]	validation_0-rmse:0.29742	validation_1-rmse:0.44433


    [24]	validation_0-rmse:0.29553	validation_1-rmse:0.44527


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
    Direct XGBoost       0.1946       0.1126       0.9449      
    OpenSTEF XGBoost     0.2049       0.1179       0.9389      



    
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
      - Direct XGBoost RMSE improvement over baseline: -7.22%
      - OpenSTEF XGBoost RMSE improvement over baseline: -12.84%


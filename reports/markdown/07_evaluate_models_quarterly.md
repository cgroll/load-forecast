    "Evaluate Combined Quarterly Model Training Results.\n\nThis report evaluates a single combined model trained on all quarters' training data\nand tested on each quarter's test period separately.\n\nThe report:\n1. Loads training results from models/quarterly_split/training_results.json\n2. Recreates predictions for visualization\n3. Creates comprehensive visualizations:\n   - Overall metrics (all test data combined)\n   - Time series plots for each quarter's test period\n   - Scatter plots for overall performance\n   - Per-quarter metrics comparison\n4. Exports metrics to metrics/quarterly_split_evaluation.json for DVC tracking\n\nThis script is designed to be run via generate_report.sh to produce HTML and Markdown outputs.\nUses Jupyter cell blocks (# %%) for interactive execution.\n"



    ======================================================================
    QUARTERLY MODEL EVALUATION REPORT
    ======================================================================
    Experiment: quarterly_split


    
    Loading results from: /home/chris/research/load-forecast/models/quarterly_split/training_results.json
    Loaded results for combined model
    Training timestamp: 2025-12-23T11:41:26.576304
    Description: Combined model trained on all quarters, evaluated on each quarter separately
    Quarters evaluated: 4


    
    ======================================================================
    EXPERIMENT CONFIGURATION
    ======================================================================
    Test days per quarter: 14
    Minimum data coverage: 95.0%
    Random seed: 42
    Training period: 2023-01-15 00:15:00 to 2023-12-17 00:00:00
    Test period: 2023-03-16 00:00:00 to 2023-12-31 23:00:00


    
    ======================================================================
    LOADING DATA FOR VISUALIZATION
    ======================================================================


    Loaded data shape: (31705, 133)
    Full data shape (with NaN): (35041, 133)


    
    ======================================================================
    RECREATING COMBINED MODEL PREDICTIONS
    ======================================================================
    Training one model on ALL quarters, testing on each quarter separately


    
    Preparing data splits for 4 quarters...
      Q1 2023: 5437 train, 1344 test
      Q2 2023: 5786 train, 1344 test
      Q3 2023: 7393 train, 1344 test
      Q4 2023: 7227 train, 1341 test
    
    Combined training data: 25843 rows
    Training combined models...
      - Baseline...
      - Direct XGBoost...


      - OpenSTEF XGBoost...


    [0]	validation_0-rmse:1.35744	validation_1-rmse:1.47545


    [1]	validation_0-rmse:0.99237	validation_1-rmse:1.10299


    [2]	validation_0-rmse:0.74694	validation_1-rmse:0.84939


    [3]	validation_0-rmse:0.58676	validation_1-rmse:0.68989


    [4]	validation_0-rmse:0.48592	validation_1-rmse:0.58991


    [5]	validation_0-rmse:0.42347	validation_1-rmse:0.52755


    [6]	validation_0-rmse:0.38533	validation_1-rmse:0.49358


    [7]	validation_0-rmse:0.36171	validation_1-rmse:0.47293


    [8]	validation_0-rmse:0.34770	validation_1-rmse:0.46215


    [9]	validation_0-rmse:0.33662	validation_1-rmse:0.45568


    [10]	validation_0-rmse:0.32998	validation_1-rmse:0.45053


    [11]	validation_0-rmse:0.32446	validation_1-rmse:0.44924


    [12]	validation_0-rmse:0.31969	validation_1-rmse:0.44766


    [13]	validation_0-rmse:0.31586	validation_1-rmse:0.44636


    [14]	validation_0-rmse:0.31150	validation_1-rmse:0.44607


    [15]	validation_0-rmse:0.30637	validation_1-rmse:0.44584


    [16]	validation_0-rmse:0.30343	validation_1-rmse:0.44650


    [17]	validation_0-rmse:0.29956	validation_1-rmse:0.44645


    [18]	validation_0-rmse:0.29681	validation_1-rmse:0.44680


    [19]	validation_0-rmse:0.29472	validation_1-rmse:0.44741


    [20]	validation_0-rmse:0.29094	validation_1-rmse:0.44736


    [21]	validation_0-rmse:0.28768	validation_1-rmse:0.44728


    [22]	validation_0-rmse:0.28582	validation_1-rmse:0.44770


    [23]	validation_0-rmse:0.28331	validation_1-rmse:0.44789


    [24]	validation_0-rmse:0.27971	validation_1-rmse:0.44775


    [25]	validation_0-rmse:0.27668	validation_1-rmse:0.44900


    
    Generating predictions for each quarter's test set...
      Q1 2023...
      Q2 2023...
      Q3 2023...


      Q4 2023...


    Predictions recreated for all quarters


    
    ======================================================================
    ORGANIZING METRICS
    ======================================================================
    Organized metrics for 4 quarters


    
    ======================================================================
    OVERALL METRICS (ALL TEST DATA COMBINED)
    ======================================================================
    
    Model              RMSE         MAE          R²          
    ------------------------------------------------------
    baseline           0.3903       0.2095       0.9557      
    direct_xgb         0.3788       0.1990       0.9582      
    openstef_xgb       0.3892       0.2044       0.9559      



    
![png](07_evaluate_models_quarterly_files/07_evaluate_models_quarterly_9_0.png)
    


    
    ======================================================================
    TIME SERIES VISUALIZATIONS
    ======================================================================



    
![png](07_evaluate_models_quarterly_files/07_evaluate_models_quarterly_10_1.png)
    


    
    ======================================================================
    SCATTER PLOTS (OVERALL - ALL TEST DATA)
    ======================================================================



    
![png](07_evaluate_models_quarterly_files/07_evaluate_models_quarterly_11_1.png)
    


    
    ======================================================================
    PER-QUARTER METRICS BREAKDOWN
    ======================================================================
    
    Quarter         Model              RMSE         MAE          R²          
    ---------------------------------------------------------------------------
    Q1 2023         baseline           0.4823       0.2502       0.9096      
    Q1 2023         direct_xgb         0.4587       0.2322       0.9182      
    Q1 2023         openstef_xgb       0.4726       0.2378       0.9132      
    ---------------------------------------------------------------------------
    Q2 2023         baseline           0.4159       0.2504       0.9682      
    Q2 2023         direct_xgb         0.3991       0.2306       0.9707      
    Q2 2023         openstef_xgb       0.4031       0.2327       0.9702      
    ---------------------------------------------------------------------------
    Q3 2023         baseline           0.4076       0.2256       0.9069      
    Q3 2023         direct_xgb         0.3993       0.2147       0.9107      
    Q3 2023         openstef_xgb       0.4183       0.2214       0.9020      
    ---------------------------------------------------------------------------
    Q4 2023         baseline           0.1928       0.1116       0.9559      
    Q4 2023         direct_xgb         0.2115       0.1184       0.9469      
    Q4 2023         openstef_xgb       0.2120       0.1256       0.9467      
    ---------------------------------------------------------------------------



    
![png](07_evaluate_models_quarterly_files/07_evaluate_models_quarterly_13_0.png)
    


    
    ======================================================================
    EXPORTING METRICS FOR DVC TRACKING
    ======================================================================
    ✓ Metrics saved to: metrics/quarterly_split_evaluation.json


    
    ======================================================================
    EVALUATION REPORT COMPLETE
    ======================================================================
    
    ✓ Analyzed 4 quarters
    ✓ Compared 3 models: Baseline, Direct XGBoost, OpenSTEF XGBoost
    ✓ Metrics exported to: metrics/quarterly_split_evaluation.json
    
    Overall Results (All Test Data Combined):
      BASELINE: RMSE=0.3903, MAE=0.2095, R²=0.9557
      DIRECT_XGB: RMSE=0.3788, MAE=0.1990, R²=0.9582
      OPENSTEF_XGB: RMSE=0.3892, MAE=0.2044, R²=0.9559


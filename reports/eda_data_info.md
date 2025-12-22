    'Exploratory Data Analysis - Part 1: Data Information and Missing Data Analysis.'



    ======================================================================
    LOADING DATA
    ======================================================================


    Raw data loaded: (35041, 44)
    Raw date range: 2023-01-01 to 2024-01-01


    
    ======================================================================
    RESAMPLING TO 15-MINUTE INTERVALS
    ======================================================================
    Data is already at 15-minute intervals. No resampling needed.


    
    ======================================================================
    DATA INFORMATION SUMMARY
    ======================================================================
    
    Number of observations: 35,041
    Date range: 2023-01-01 to 2024-01-01 (365 days)
    
    Target variable: 'load'
      - Missing values: 22 (0.06%)
    
    Feature variables (X):
      - Number of variables: 43
      - Observations with any missing feature: 1,649 (4.71%)
    
    Complete observations (X + y both present): 33,370 (95.23%)


    
    ======================================================================
    MISSING DATA GAP ANALYSIS
    ======================================================================
    
    Target (load):
    ----------------------------------------------------------------------
      Total gaps: 3
      Total missing observations: 22
      Average gap size: 7.3 observations
      Median gap size: 5 observations
      Largest gap: 16 observations (0.16 days)
    
      Top 10 Largest Gaps:
      #     Start Date      End Date        Days       Obs       
      ------------------------------------------------------------
      1     2023-05-23      2023-05-23      0.16       16        
      2     2023-10-29      2023-10-29      0.04       5         
      3     2023-01-01      2023-01-01      0.00       1         
    
    Features (X):
    ----------------------------------------------------------------------
      Total gaps: 4
      Total missing observations: 1,649
      Average gap size: 412.2 observations
      Median gap size: 155 observations
      Largest gap: 1,335 observations (13.90 days)
    
      Top 10 Largest Gaps:
      #     Start Date      End Date        Days       Obs       
      ------------------------------------------------------------
      1     2023-05-29      2023-06-12      13.90      1335      
      2     2023-01-29      2023-01-31      2.35       227       
      3     2023-03-22      2023-03-23      0.85       83        
      4     2023-12-31      2024-01-01      0.03       4         
    
    Overall (any):
    ----------------------------------------------------------------------
      Total gaps: 7
      Total missing observations: 1,671
      Average gap size: 238.7 observations
      Median gap size: 16 observations
      Largest gap: 1,335 observations (13.90 days)
    
      Top 10 Largest Gaps:
      #     Start Date      End Date        Days       Obs       
      ------------------------------------------------------------
      1     2023-05-29      2023-06-12      13.90      1335      
      2     2023-01-29      2023-01-31      2.35       227       
      3     2023-03-22      2023-03-23      0.85       83        
      4     2023-05-23      2023-05-23      0.16       16        
      5     2023-10-29      2023-10-29      0.04       5         
      6     2023-12-31      2024-01-01      0.03       4         
      7     2023-01-01      2023-01-01      0.00       1         



    
![png](eda_data_info_files/eda_data_info_8_0.png)
    


    
    ======================================================================
    VARIABLE CATEGORIZATION
    ======================================================================
    
    Load variables (1): ['load']
    
    Weather/Climate variables (12):
      - clearSky_dlf
      - clearSky_ulf
      - clouds
      - humidity
      - mxlD
      - pressure
      - radiation
      - snowDepth
      - temp
      - winddeg
      - windspeed
      - windspeed_100m
    
    Pricing variables (1): ['APX']
    
    Load profile variables (30):
      - E1A_AMI_A
      - E1A_AMI_I
      - E1A_AZI_A
      - E1A_AZI_I
      - E1B_AMI_A
      - E1B_AMI_I
      - E1B_AZI_A
      - E1B_AZI_I
      - E1C_AMI_A
      - E1C_AMI_I
      - E1C_AZI_A
      - E1C_AZI_I
      - E2A_AMI_A
      - E2A_AMI_I
      - E2A_AZI_A
      - E2A_AZI_I
      - E2B_AMI_A
      - E2B_AMI_I
      - E2B_AZI_A
      - E2B_AZI_I
      - E3A_A
      - E3A_I
      - E3B_A
      - E3B_I
      - E3C_A
      - E3C_I
      - E3D_A
      - E3D_I
      - E4A_A
      - E4A_I


    
    ======================================================================
    EDA PART 1 (DATA INFO) COMPLETE
    ======================================================================


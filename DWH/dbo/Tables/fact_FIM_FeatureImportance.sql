CREATE TABLE [dbo].[fact_FIM_FeatureImportance] (
    [FIM_ID]          INT            IDENTITY (1, 1) NOT NULL,
    [FIM_TRMS_ID]     INT            NOT NULL,
    [FIM_FeatureName] NVARCHAR (MAX) NOT NULL,
    [FIM_Value_1]     FLOAT (53)     NULL,
    [FIM_Value_2]     FLOAT (53)     NULL,
    PRIMARY KEY CLUSTERED ([FIM_ID] ASC)
);


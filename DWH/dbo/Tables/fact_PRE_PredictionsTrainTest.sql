CREATE TABLE [dbo].[fact_PRE_PredictionsTrainTest] (
    [PRE_ID]               INT        IDENTITY (1, 1) NOT NULL,
    [PRE_TDS_ID]           INT        NULL,
    [PRE_TRMS_ID]          INT        NULL,
    [PRE_OriginalValue_1]  FLOAT (53) NULL,
    [PRE_PredictedValue_1] FLOAT (53) NOT NULL,
    [PRE_OriginalValue_2]  FLOAT (53) NULL,
    [PRE_PredictedValue_2] FLOAT (53) NOT NULL,
    [PRE_TYP_ID]           INT        NOT NULL
);


CREATE TABLE [dbo].[fact_FEA_Features] (
    [FEA_ID]          INT            IDENTITY (1, 1) NOT NULL,
    [FEA_PES_ID]      INT            NOT NULL,
    [FEA_FeatureName] NVARCHAR (MAX) NOT NULL,
    [FEA_Index]       INT            NOT NULL,
    [FEA_F_value]     FLOAT (53)     NULL,
    [FEA_p_value]     FLOAT (53)     NULL,
    PRIMARY KEY CLUSTERED ([FEA_ID] ASC)
);


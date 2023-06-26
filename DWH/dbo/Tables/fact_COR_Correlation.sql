CREATE TABLE [dbo].[fact_COR_Correlation] (
    [COR_ID]            INT            IDENTITY (1, 1) NOT NULL,
    [COR_PES_ID]        INT            NOT NULL,
    [COR_FeatureName_1] NVARCHAR (MAX) NOT NULL,
    [COR_FeatureName_2] NVARCHAR (MAX) NOT NULL,
    [COR_Value]         FLOAT (53)     NOT NULL,
    PRIMARY KEY CLUSTERED ([COR_ID] ASC)
);


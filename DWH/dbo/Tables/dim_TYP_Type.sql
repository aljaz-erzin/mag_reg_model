﻿CREATE TABLE [dbo].[dim_TYP_Type] (
    [TYP_ID]   INT           IDENTITY (1, 1) NOT NULL,
    [TYP_Name] NVARCHAR (50) NOT NULL,
    PRIMARY KEY CLUSTERED ([TYP_ID] ASC)
);


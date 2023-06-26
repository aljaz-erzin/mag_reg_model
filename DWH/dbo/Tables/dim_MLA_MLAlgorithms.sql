CREATE TABLE [dbo].[dim_MLA_MLAlgorithms] (
    [MLA_ID]            INT            IDENTITY (1, 1) NOT NULL,
    [MLA_Name]          NVARCHAR (MAX) NULL,
    [MLA_Description]   NVARCHAR (MAX) NULL,
    [MLA_PythonFunName] NVARCHAR (MAX) NOT NULL,
    [MLA_MultiTarget]   TINYINT        NOT NULL,
    PRIMARY KEY CLUSTERED ([MLA_ID] ASC)
);


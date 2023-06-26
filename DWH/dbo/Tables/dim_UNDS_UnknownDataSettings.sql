CREATE TABLE [dbo].[dim_UNDS_UnknownDataSettings] (
    [UNDS_ID]      INT     IDENTITY (1, 1) NOT NULL,
    [UNDS_TRMS_ID] INT     NOT NULL,
    [UNDS_Active]  TINYINT NOT NULL,
    PRIMARY KEY CLUSTERED ([UNDS_ID] ASC)
);


CREATE TABLE [dbo].[dim_ODL_Odlocbe] (
    [ID_Odlocba]          UNIQUEIDENTIFIER NULL,
    [ODL StevilkaOdlocbe] NVARCHAR (50)    NULL,
    [ODL DatumOdlocbe]    DATETIME2 (7)    NULL,
    [ODL DatumVrocitve]   DATETIME2 (7)    NULL,
    [ODL RokZaIzvedboDel] DATETIME2 (7)    NULL,
    [ODL Tip]             NVARCHAR (50)    NULL,
    [ODL Vrsta]           NVARCHAR (50)    NULL
);


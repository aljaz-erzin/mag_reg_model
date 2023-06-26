﻿CREATE TABLE [dbo].[dim_PES_PythonExecSettings] (
    [PES_ID]                          INT            IDENTITY (1, 1) NOT NULL,
    [PES_Table]                       NVARCHAR (50)  NOT NULL,
    [PES_Active]                      TINYINT        NOT NULL,
    [PES_TargetVariables]             NVARCHAR (MAX) NOT NULL,
    [PES_MultiTarget]                 TINYINT        NOT NULL,
    [PES_OneHotEncodeFeatures]        NVARCHAR (MAX) NULL,
    [PES_ContinousFeatures]           NVARCHAR (MAX) NULL,
    [PES_ExcludedFeatures]            NVARCHAR (MAX) NULL,
    [PES_DataSetDescription]          NVARCHAR (MAX) NULL,
    [PES_Test_Portion]                FLOAT (53)     NOT NULL,
    [PES_CrossValidation]             TINYINT        NOT NULL,
    [PES_CrossValidation_k]           INT            NULL,
    [PES_Table_Unknown]               NVARCHAR (MAX) NULL,
    [PES_feat_corr_threshold]         FLOAT (53)     DEFAULT ((0.9)) NULL,
    [PES_left_border]                 FLOAT (53)     NULL,
    [PES_right_border]                FLOAT (53)     NULL,
    [PES_add_weights]                 TINYINT        NULL,
    [PES_p_threshold]                 FLOAT (53)     DEFAULT ((0.0)) NULL,
    [PES_exclude_gt_p]                TINYINT        DEFAULT ((0)) NULL,
    [PES_PCA_variance_kept]           FLOAT (53)     DEFAULT ((1.0)) NULL,
    [PES_exclude_constants_threshold] FLOAT (53)     DEFAULT ((0.0)) NULL,
    [PES_DateFrom]                    DATE           DEFAULT ('1900-01-01') NULL,
    [PES_Odlocba_left_border]         FLOAT (53)     NULL,
    [PES_Odlocba_right_border]        FLOAT (53)     NULL,
    PRIMARY KEY CLUSTERED ([PES_ID] ASC)
);

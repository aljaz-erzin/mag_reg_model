from common import *
from sqlalchemy import create_engine, text
from db_connection import close_connection_to_db, connect_to_db,pes_table, pre_table, fim_table, unds_table, trms_table, mla_table, fea_table, cor_table, pre_table_unknown
from math import ceil
import pandas as pd
import numpy as np


"""
---------------------------  GROUP: SELECT  ---------------------------  
"""

def select_data(PES, left_grouped=None, right_grouped= None, left_odlocbe=None, right_odlocbe= None, left_kolicina = None, right_kolicina = None, left_kubatura = None, right_kubatura = None, target_vars =None):
    table = PES['PES_Table']
    datefrom = PES['PES_DateFrom']
    if target_vars is None:
        return None
    target_string = ' FCT.' + ' + FCT.'.join(target_vars)
    #print(target_string)
    if datefrom is None:
        datefrom = '1900-01-01'
    db_conn = connect_to_db()
    if left_grouped is None:
        left_grouped = -2
    if right_grouped is None:
        right_grouped = 2
    if left_odlocbe is None:
        left_odlocbe = -2
    if right_odlocbe is None:
        right_odlocbe = 2
    if left_kolicina is None:
        left_kolicina = -9999999
    if right_kolicina is None:
        right_kolicina = 9999999
    if left_kubatura is None:
        left_kubatura = -9999999
    if right_kubatura is None:
        right_kubatura = 9999999
    query = 'SELECT FCT.* FROM ' + table + ' FCT  INNER JOIN dbo.dim_ODL_Odlocbe DN ON DN.[ID_Odlocba] = FCT.[ID_Odlocba] INNER JOIN [dbo].[dim_Odlocbe_Ratio] RAT ON RAT.ID_Odlocba = FCT.[ID_Odlocba]  WHERE ('+target_string+') BETWEEN '+str(left_kolicina) + ' AND '+str(right_kolicina)+' AND  (FCT.[KubatureSUM]) BETWEEN '+str(left_kubatura) + ' AND '+str(right_kubatura)+' AND RAT.[RATIO] > ' + str(left_odlocbe) + ' AND RAT.[RATIO] < ' + str(right_odlocbe) + ' AND FCT.[RATIO] > ' + str(left_grouped) + ' AND FCT.[RATIO] < ' + str(right_grouped) + " AND CONVERT(DATE, DN.[ODL DatumOdlocbe]) >= '%s'"% str(datefrom)
    res = pd.DataFrame(db_conn.execute(text(query)))
    close_connection_to_db(db_conn)
    return res

def PES_select_active_ids():
    db_conn = connect_to_db()
    query = 'SELECT * FROM ' + pes_table + ' WHERE [PES_Active] = 1'
    res = pd.DataFrame(db_conn.execute(text(query)))
    close_connection_to_db(db_conn)
    return res



def RPEDPLOT_get_trms_id(table):
    db_conn = connect_to_db()
    query = 'SELECT TOP(1) [PRE_TRMS_ID] FROM ' + table
    res = pd.DataFrame(db_conn.execute(text(query)))
    close_connection_to_db(db_conn)
    return res

def PREDPLOT_get_column(table, column, type = None):
    db_conn = connect_to_db()
    if type is None:
        query = 'SELECT [' +column + '] FROM ' + table + ' ORDER BY [PRE_ID] ASC'
    else:
        query = 'SELECT [' +column + '] FROM ' + table + " WHERE [TYP_Name] = '%s' ORDER BY [PRE_ID] ASC" % type
    res = pd.DataFrame(db_conn.execute(text(query)))
    close_connection_to_db(db_conn)
    return res

def DIS_get_column(table, column, datefrom, type):
    db_conn = connect_to_db()
    print(datefrom)
    query = 'SELECT [' +column + '] FROM ' + table +  " FCT INNER JOIN dbo.dim_ODL_Odlocbe DN ON DN.[ID_Odlocba] = FCT.[ID_Odlocba] WHERE CONVERT(DATE, DN.[ODL DatumOdlocbe]) >= '%s'"% str(datefrom)
    res = pd.DataFrame(db_conn.execute(text(query)))
    close_connection_to_db(db_conn)
    return res


def UNDS_select_active_ids():
    query = 'SELECT * FROM ' + unds_table + ' WHERE [UNDS_Active] = 1'
    db_conn = connect_to_db()
    res = pd.DataFrame(db_conn.execute(text(query)))
    close_connection_to_db(db_conn)
    return res

def TRMS_select_active_by_pesid(PES_ID):
    query = 'SELECT * FROM ' + trms_table + ' WHERE [TRMS_Active] = 1 AND [TRMS_PES_ID] = %d' % PES_ID
    db_conn = connect_to_db()
    res = pd.DataFrame(db_conn.execute(text(query)))
    close_connection_to_db(db_conn)
    return res

def TRMS_select_by_id(TRMS_ID):
    query = 'SELECT * FROM ' + trms_table + ' WHERE [TRMS_ID] = %d' % TRMS_ID
    db_conn = connect_to_db()
    res = pd.DataFrame(db_conn.execute(text(query)))
    close_connection_to_db(db_conn)
    return res

def PES_select_by_id(PES_ID):
    query = 'SELECT * FROM ' + pes_table + ' WHERE [PES_ID] = %d' % PES_ID
    db_conn = connect_to_db()
    res = pd.DataFrame(db_conn.execute(text(query)))
    close_connection_to_db(db_conn)
    return res

def FEA_select_by_pes_id(PES_ID):
    query = 'SELECT [FEA_FeatureName] FROM ' + fea_table + ' WHERE [FEA_PES_ID] = %d ORDER BY [FEA_Index] ASC' % PES_ID
    db_conn = connect_to_db()
    res = pd.DataFrame(db_conn.execute(text(query)))
    close_connection_to_db(db_conn)
    return res

def MLA_select_by_id(MLA_ID, multitarget):
    query = 'SELECT [MLA_PythonFunName], [MLA_MultiTarget] FROM ' + mla_table + ' WHERE [MLA_ID] = %d AND [MLA_MultiTarget] = %d' % (MLA_ID, multitarget)
    db_conn = connect_to_db()
    res = pd.DataFrame(db_conn.execute(text(query)))
    close_connection_to_db(db_conn)
    return res








"""
---------------------------  GROUP: INSERT  ---------------------------  
"""

def insert_in_batches(table, data, columns, db_conn):
    for i in range(1,ceil(len(data)/1000)+1):
        batch_data = ', '.join(map(str, data[(i-1)*1000:i*1000]))
        sql = "INSERT INTO "+ table +" ("+columns+") VALUES {}".format(batch_data)
        db_conn.execute(text(sql))
        db_conn.commit()

def COR_insert_correlations(data, PES_ID):
    db_conn = connect_to_db()
    query = 'DELETE FROM ' + cor_table + ' WHERE [COR_PES_ID] = %d' % PES_ID
    db_conn.execute(text(query))
    db_conn.commit()
    insert_in_batches(cor_table, data, "[COR_PES_ID], [COR_FeatureName_1], [COR_FeatureName_2], [COR_Value]", db_conn)
    close_connection_to_db(db_conn)

def FEA_insert_data(data, feat_tar_corr, PES_ID):
    db_conn = connect_to_db()
    query = 'DELETE FROM ' + fea_table + ' WHERE [FEA_PES_ID] = %d' % PES_ID
    db_conn.execute(text(query))
    db_conn.commit()
    new_data = []
    for f in data:
        flag = False
        for d in feat_tar_corr:
            if d['feature'] == f:
                new_data.append({'feature':f, 'F-value':d['F-value'], 'p-value':d['p-value']})
                flag = True
                break
        if not flag:
            new_data.append({'feature':f, 'F-value':None, 'p-value':None})
    data = np.array(new_data)
    data = [(PES_ID, d['feature'], (i+1), d['F-value'], d['p-value']) for (i, d) in enumerate(data)]
    insert_in_batches(fea_table, data, "[FEA_PES_ID], [FEA_FeatureName], [FEA_Index], [FEA_F_value], [FEA_p_value]", db_conn)
    close_connection_to_db(db_conn)

def FIM_insert_data(features, values, TRMS_ID):
    db_conn = connect_to_db()
    query = 'DELETE FROM ' + fim_table + ' WHERE [FIM_TRMS_ID] = %d' % TRMS_ID
    db_conn.execute(text(query))
    db_conn.commit()
    if values.shape[1] == 2:
        data = [(TRMS_ID, f, values[i, 0], values[i, 1]) for (i, f) in enumerate(features)]
        insert_in_batches(fim_table, data, "[FIM_TRMS_ID], [FIM_FeatureName], [FIM_Value_1], [FIM_Value_2]", db_conn)
    if values.shape[1] == 1:
        data = [(TRMS_ID, f, values[i, 0]) for (i, f) in enumerate(features)]
        insert_in_batches(fim_table, data, "[FIM_TRMS_ID], [FIM_FeatureName], [FIM_Value_1]", db_conn)
    close_connection_to_db(db_conn)

def PRE_insert_multitarget_data(TDS_IDS, original, predicted, TRMS_ID, type):
    db_conn = connect_to_db()
    if type == 3:
        query = "DELETE FROM " + pre_table_unknown + " WHERE [PRE_TRMS_ID] = %d AND [PRE_TYP_ID] = '%d'" % (TRMS_ID, type)
    else:
        query = "DELETE FROM " + pre_table + " WHERE [PRE_TRMS_ID] = %d AND [PRE_TYP_ID] = '%d'" % (TRMS_ID, type)
    db_conn.execute(text(query))
    db_conn.commit()
    if type == 3:
        data = [(TDS_ID, TRMS_ID, predicted[i][0], predicted[i][1], type) for
                (i, TDS_ID) in enumerate(TDS_IDS)]
        insert_in_batches(pre_table_unknown, data,
                          "[PRE_UDS_ID], [PRE_TRMS_ID], [PRE_PredictedValue_1], [PRE_PredictedValue_2], [PRE_TYP_ID]", db_conn)
    else:
        data = [(TDS_ID, TRMS_ID, float(original[i][0]), predicted[i][0], float(original[i][1]), predicted[i][1], type) for (i, TDS_ID) in enumerate(TDS_IDS)]
        insert_in_batches(pre_table, data, "[PRE_TDS_ID], [PRE_TRMS_ID], [PRE_OriginalValue_1], [PRE_PredictedValue_1], [PRE_OriginalValue_2], [PRE_PredictedValue_2], [PRE_TYP_ID]", db_conn)
    close_connection_to_db(db_conn)

def PRE_insert_single_target_data(TDS_IDS, original, predicted, TRMS_ID, type):
    db_conn = connect_to_db()
    if type == 3:
        query = "DELETE FROM " + pre_table_unknown + " WHERE [PRE_TRMS_ID] = %d AND [PRE_TYP_ID] = '%d'" % (TRMS_ID, type)
    else:
        query = "DELETE FROM " + pre_table + " WHERE [PRE_TRMS_ID] = %d AND [PRE_TYP_ID] = '%d'" % (TRMS_ID, type)
    db_conn.execute(text(query))
    db_conn.commit()
    if type == 3:
        data = [(TDS_ID, TRMS_ID, predicted[i], type) for
                (i, TDS_ID) in enumerate(TDS_IDS)]
        insert_in_batches(pre_table, data,
                          "[PRE_UDS_ID], [PRE_TRMS_ID], [PRE_PredictedValue_1], [PRE_TYP_ID]", db_conn)
    else:
        data = [(TDS_ID, TRMS_ID, float(original[i]), predicted[i],  type)
                for (i, TDS_ID) in enumerate(TDS_IDS)]
        insert_in_batches(pre_table, data,
                          "[PRE_TDS_ID], [PRE_TRMS_ID], [PRE_OriginalValue_1], [PRE_PredictedValue_1], [PRE_TYP_ID]", db_conn)
    close_connection_to_db(db_conn)

def TRMS_update_eval_scores(TRMS, r2_scores, rmse_scores, r2_scores_all, rmse_scores_all, PES_TargetVariables, type):
    TRMS_ID = TRMS['TRMS_ID']
    db_conn = connect_to_db()
    if len(PES_TargetVariables) > 1:
        data = "[TRMS_"+type+"_R2_1] = "+str(r2_scores[0])+", [TRMS_"+type+"_RMSE_1] = "+str(rmse_scores[0])+\
               ", [TRMS_"+type+"_R2_2] = "+str(r2_scores[1])+", [TRMS_"+type+"_RMSE_2] = "+str(rmse_scores[1])+\
               ", [TRMS_"+type+"_R2] = "+str(r2_scores_all)+", [TRMS_"+type+"_RMSE] = "+str(rmse_scores_all)+\
                ", [TRMS_AVG_"+type+"_R2_1] = "+str((TRMS['TRMS_N_Executions'] * TRMS["TRMS_AVG_"+type+"_R2_1"] + r2_scores[0]) / (TRMS['TRMS_N_Executions'] + 1))+", [TRMS_AVG_"+type+"_RMSE_1] = "+str((TRMS['TRMS_N_Executions'] * TRMS["TRMS_AVG_"+type+"_RMSE_1"] + rmse_scores[0]) / (TRMS['TRMS_N_Executions'] + 1))+\
               ", [TRMS_AVG_"+type+"_R2_2] = "+str((TRMS['TRMS_N_Executions'] * TRMS["TRMS_AVG_"+type+"_R2_2"] + r2_scores[1]) / (TRMS['TRMS_N_Executions'] + 1))+", [TRMS_AVG_"+type+"_RMSE_2] = "+str((TRMS['TRMS_N_Executions'] * TRMS["TRMS_AVG_"+type+"_RMSE_2"] + rmse_scores[1]) / (TRMS['TRMS_N_Executions'] + 1))+\
               ", [TRMS_AVG_"+type+"_R2] = "+str((TRMS['TRMS_N_Executions'] * TRMS["TRMS_AVG_"+type+"_R2"] + r2_scores_all) / (TRMS['TRMS_N_Executions'] + 1))+", [TRMS_AVG_"+type+"_RMSE] = "+str((TRMS['TRMS_N_Executions'] * TRMS["TRMS_AVG_"+type+"_RMSE"] + rmse_scores_all) / (TRMS['TRMS_N_Executions'] + 1))+\
               ", [TRMS_N_Executions] = "+str(TRMS['TRMS_N_Executions'] + 1) + ", [TRMS_Target_1] = '"+str(PES_TargetVariables[0]) + "', [TRMS_Target_2] = '"+str(PES_TargetVariables[1]) + "'"
    else:
        data = "[TRMS_"+type+"_R2_1] = "+str(r2_scores[0])+", [TRMS_"+type+"_RMSE_1] = "+str(rmse_scores[0])+\
               ", [TRMS_"+type+"_R2] = "+str(r2_scores_all)+", [TRMS_"+type+"_RMSE] = "+str(rmse_scores_all)+\
                ", [TRMS_AVG_"+type+"_R2_1] = "+str((TRMS['TRMS_N_Executions'] * TRMS["TRMS_AVG_"+type+"_R2_1"] + r2_scores[0]) / (TRMS['TRMS_N_Executions'] + 1))+", [TRMS_AVG_"+type+"_RMSE_1] = "+str((TRMS['TRMS_N_Executions'] * TRMS["TRMS_AVG_"+type+"_RMSE_1"] + rmse_scores[0]) / (TRMS['TRMS_N_Executions'] + 1))+\
               ", [TRMS_AVG_"+type+"_R2] = "+str((TRMS['TRMS_N_Executions'] * TRMS["TRMS_AVG_"+type+"_R2"] + r2_scores_all) / (TRMS['TRMS_N_Executions'] + 1))+", [TRMS_AVG_"+type+"_RMSE] = "+str((TRMS['TRMS_N_Executions'] * TRMS["TRMS_AVG_"+type+"_RMSE"] + rmse_scores_all) / (TRMS['TRMS_N_Executions'] + 1))+\
               ", [TRMS_Target_1] = '"+str(PES_TargetVariables[0]) + "'"
    query = 'UPDATE  ' + trms_table + ' SET '+data+' WHERE [TRMS_ID] = %d' % TRMS_ID
    db_conn.execute(text(query))
    db_conn.commit()
    close_connection_to_db(db_conn)



def TDS_update_weights(data, pes_table):
    print("UPDATING WEIGHTS")
    db_conn = connect_to_db()
    for i in range(len(data)):
        sql = "UPDATE " + pes_table + " SET  [TRAIN_WEIGHT] = " + str(data[i][1]) + " WHERE [ID] = " + str(data[i][0])
        db_conn.execute(text(sql))
    db_conn.commit()
    close_connection_to_db(db_conn)
    print("DONE UPDATING WEIGHTS")

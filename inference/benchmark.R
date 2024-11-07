rm(list = ls())

setwd('C:/Active/projects/iDriver/iDriver/')

library(data.table)
library(openxlsx)
library(dplyr)
library(pROC)
library(PRROC)

source("benchmark/functions_benchmark.R")
source("benchmark/functions_annotate_elements.R")

path_procPCAWG_res <- "../extdata/procInput/PCAWG_results/"
path_proccessedGE <- "../extdata/procInput/"
tissue <-  "Pancan-no-skin-melanoma-lymph"
path_save_benchRes <- "../../make_features/external/BMR/benchmark_.05/"

########### 1) load the pRes object and annotated PCAWG_IDs ###########
ann_PCAWG_ID_complement <- fread(paste0(path_proccessedGE, "ann_PCAWG_ID_complement.csv"))
load(paste0(path_procPCAWG_res, tissue, ".RData"))


########### 2) add new result(s) to pRes for computing measure stats and saving the measurment results ##########
newRESULTS <- c("GBM", 
                 "RF", "nn_mseLoss", "nn_poisLoss",
                "eMET")
# PATHs_newResults <- c("../../make_features/external/BMR/output/with_RepliSeq_HiC/bin_size_effect/var_size/GBM/inference/GBM_inference.tsv",
#                       "../../make_features/external/BMR/output/with_RepliSeq_HiC/bin_size_effect/var_size/RF/inference/RF_inference.tsv",
#                       "../../make_features/external/BMR/output/with_RepliSeq_HiC/bin_size_effect/var_size/nn_mseLoss/inference/nn_mseLoss_inference.tsv",
#                       "../../make_features/external/BMR/output/with_RepliSeq_HiC/bin_size_effect/var_size/nn_poisLoss/inference/nn_poisLoss_inference.tsv",
                      # "../../make_features/external/BMR/output/TL/GBM/inference/GBM_inference.tsv")

PATHs_newResults <- c("../../make_features/external/BMR/output/with_RepliSeq_HiC/bin_size_effect/var_size/GBM/inference/GBM_inference_binomTest.tsv",
                      "../../make_features/external/BMR/output/with_RepliSeq_HiC/bin_size_effect/var_size/RF/inference/RF_inference_binomTest.tsv",
                      "../../make_features/external/BMR/output/with_RepliSeq_HiC/bin_size_effect/var_size/nn_mseLoss/inference/nn_mseLoss_inference_binomTest.tsv",
                      "../../make_features/external/BMR/output/with_RepliSeq_HiC/bin_size_effect/var_size/nn_poisLoss/inference/nn_poisLoss_inference_binomTest.tsv",
                      "../../make_features/external/BMR/output/TL/GBM/inference/GBM_inference_binomTest.tsv")



sig_definition_methods <- c("fdr")
sig_definition_thresholds <- c(.05)

add_newMethod_to_pRes <- function(pRes, path_newRes){
  
  newMethod <- data.frame(fread(path_newRes))
  newMethod <- newMethod[which(!grepl("lncrna.ncrna", newMethod$binID)),]
  newMethod <- newMethod[which(!grepl("lncrna.promCore", newMethod$binID)),]
  # newMethod <- newMethod[which(newMethod$nSample > 1),]
  idx_pVal <- which(colnames(newMethod) %in% c("pValues", "p_value",
                                               "raw_p", "pvals", 
                                               "raw_p_binom" )) #  "raw_p_nBinom"
  idx_ID <- which(colnames(newMethod) %in% c("binID", "PCAWG_ID"))
  
  newMethod <- data.frame("PCAWG_IDs" = newMethod[,idx_ID], 
                          "p_value" =  newMethod[,idx_pVal],
                          "q_value" = rep(NA, nrow(newMethod)),
                          "fdr" =  p.adjust(newMethod[,idx_pVal], "fdr"))
  
  newMethod <- newMethod[which(!is.na(newMethod$p_value)),]
  
  pRes$newMethod <- newMethod
  class(pRes) = "pRes"
  pRes
}



save_Measures_inSingletable <- function(path_save, newRESULTS, PATHs_newResults, pRes,
                                        sig_definition_methods,
                                        sig_definition_thresholds,
                                        ann_PCAWG_ID_complement, tissue, based_on,
                                        compareRank_new = list(FALSE, NA)){

  if (length(PATHs_newResults) != 0){
    for (i in 1:length(PATHs_newResults)) {
      pRes <- add_newMethod_to_pRes(pRes, PATHs_newResults[i])
      names(pRes) <- c(names(pRes)[1:(length(pRes) -1)], newRESULTS[i])
    }
  }
  
  pRes <- pRes[10:14]



  for (j in 1:length(sig_definition_methods)) {
    sigCriteria <- define_significant_criteria(sig_definition_methods[j],
                                               sig_definition_thresholds[j])

    st1 <- computeStat_1(pRes, sigCriteria, ann_PCAWG_ID_complement, based_on)
    st_2 <- computeStat_2(pRes, sigCriteria, ann_PCAWG_ID_complement, based_on)$tables

    CDS_tab1 <- cbind(st1$CDS, "method" = rownames(st1$CDS))
    CDS <- full_join(CDS_tab1, st_2$CDS)

    NC_tab1 <- cbind(st1$non_coding, "method" = rownames(st1$non_coding))
    non_coding <- full_join(NC_tab1, st_2$non_coding)

    tab_all <- full_join(CDS, non_coding, by = "method")

    colnames(CDS) <- gsub("CDS_", "", colnames(CDS))
    colnames(non_coding) <- gsub("NC_", "", colnames(non_coding))



    Measures <- list("CDS" = CDS, "non_coding" = non_coding,
                     "all" = tab_all)

    df <- tab_all
    col_order <- c("method", "CDS_nTPs", "CDS_nHits",
                   "CDS_PRECISIONs", "CDS_Recalls", "CDS_F1", "CDS_AUC",
                   "CDS_AUPR", "CDS_n_elemnts", "nc_n_elemnts", "NC_nTPs",
                   "NC_nHits", "NC_PRECISIONs", "NC_Recalls", "NC_F1",
                   "NC_AUC", "NC_AUPR")

    if (sig_definition_methods[j] == "fdr" & compareRank_new[[1]]) {
      compare_nTPs <- c()
      for(k in 1:nrow(df)){
        sigCr_cds <- define_significant_criteria("fixedNumberOfElems",
                                                 df$CDS_nHits[k] )
        st1_nTP_cds <- computeStat_1(pRes, sigCr_cds, ann_PCAWG_ID_complement, based_on)
        sigCr_nc <- define_significant_criteria("fixedNumberOfElems",
                                                df$NC_nHits[k] )
        st1_nTP_nc <- computeStat_1(pRes, sigCr_nc, ann_PCAWG_ID_complement, based_on)

        compare_nTPs <- rbind(compare_nTPs,
                              c(st1_nTP_cds$CDS[compareRank_new[[2]], "CDS_nTPs"],
                                st1_nTP_nc$non_coding[compareRank_new[[2]], "NC_nTPs"]))

      }
      colnames(compare_nTPs) <- c("CDS_nTPcompare", "NC_nTPcompare")
      sigCr_cds2 <- define_significant_criteria("fixedNumberOfElems",
                                                df[which(df$method==compareRank_new[[2]]),
                                                   which(colnames(df)=="CDS_nHits")])
      st_nTP_cds <- computeStat_1(pRes, sigCr_cds2, ann_PCAWG_ID_complement, based_on)
      CDS_nTPcompare_reverse=st_nTP_cds$CDS$CDS_nTPs
      df <- cbind(df, CDS_nTPcompare_reverse, compare_nTPs)
      col_order <- c("method", "CDS_nTPs", "CDS_nHits", "CDS_nTPcompare",
                     "CDS_nTPcompare_reverse",
                     "CDS_PRECISIONs", "CDS_Recalls", "CDS_F1", "CDS_AUC",
                     "CDS_AUPR", "CDS_n_elemnts", "nc_n_elemnts", "NC_nTPs",
                     "NC_nHits", "NC_nTPcompare", "NC_PRECISIONs", "NC_Recalls",
                     "NC_F1", "NC_AUC", "NC_AUPR")
    }

    # Use the select function to reorder the columns
    df_reordered <- df %>% select(col_order)

    dir.create(paste0(path_save, tissue, "/tables/"),
               showWarnings = F,
               recursive = T)
    write.csv(df_reordered, file = paste0(path_save, tissue,
                                          "/tables/table_GoldStd_basedon_",
                                          based_on,
                                          sigCriteria$method, ".csv"))

    save(Measures, file = paste0(path_save, tissue,
                                 "/tables/Measures_GoldStd_basedon_", based_on,
                                 sigCriteria$method, ".RData"))
  }

}





## use PCAWG drivers/ CGC as gold standard for defining true positives

driver_based_on <- c("in_CGC_new", "all", "in_pcawg", "in_oncoKB", "any")

for (based_on in driver_based_on) {
  save_Measures_inSingletable(path_save_benchRes, newRESULTS, PATHs_newResults, pRes, 
                              sig_definition_methods,
                              sig_definition_thresholds,
                              ann_PCAWG_ID_complement, tissue, based_on #,
                              # compareRank_new = list(TRUE, "GBM")
  )
  
  
}


##############################################################################
if (length(PATHs_newResults) != 0){
  for (i in 1:length(PATHs_newResults)) {
    pRes <- add_newMethod_to_pRes(pRes, PATHs_newResults[i])
    names(pRes) <- c(names(pRes)[1:(length(pRes) -1)], newRESULTS[i])
  } 
}



pRes <- pRes[10:14]
based_on <- "in_oncoKB"

sigCriteria <- define_significant_criteria(sig_definition_methods,
                                           sig_definition_thresholds)

comp_pRes <- annotate_pRes(pRes, sigCriteria, ann_PCAWG_ID_complement, based_on)

all_gbm_cds <- comp_pRes[["GBM"]][["CDS"]]

gbm_cds <- all_gbm_cds[which(all_gbm_cds$fdr < .05),]
all_TL_cds <- comp_pRes[['eMET']][['CDS']]
TL_cds <- all_TL_cds[which(all_TL_cds$fdr < .05),]

dir.create('../../make_features/external/BMR/tables/', showWarnings = F, recursive = T)
fwrite(gbm_cds, file = '../../make_features/external/BMR/tables/sigHits_XGBoost.tsv', sep = '\t')
ann <- ann_PCAWG_ID_complement[,c("PCAWG_IDs", "type_of_element", "in_CGC",
                                  "in_CGC_literature", "in_CGC_new", "in_oncoKB",
                                  "in_pcawg", "tissues")]

TL_cds <- left_join(TL_cds, ann, by = 'PCAWG_IDs')
eMET_table <- TL_cds[, c("PCAWG_IDs", "p_value", "fdr", "is_TP", "type_of_element", "in_CGC",
"in_CGC_literature", "in_CGC_new", "in_oncoKB", "in_pcawg")]
fwrite(eMET_table, file = '../../make_features/external/BMR/tables/sigHits_eMET.tsv',
       sep = '\t')

which(!TL_cds$PCAWG_IDs %in% gbm_cds$PCAWG_IDs)

# 41 59 63 65 66 69 70 73 76
TL_exclusive_calls <- TL_cds$PCAWG_IDs[which(!TL_cds$PCAWG_IDs %in% gbm_cds$PCAWG_IDs)]
all_gbm_cds[which(all_gbm_cds$PCAWG_IDs %in% TL_exclusive_calls),]

# PCAWG_IDs      p_value q_value        fdr positive is_goldStandard is_TP
# 66946    gc19_pc.cds::gencode::RNF43::ENSG00000108375.8 6.622163e-05      NA 0.05268621    FALSE            TRUE FALSE
# 66557   gc19_pc.cds::gencode::RBM10::ENSG00000182872.11 7.907422e-05      NA 0.05948802    FALSE            TRUE FALSE
# 60647    gc19_pc.cds::gencode::HRAS::ENSG00000174775.12 8.289316e-05      NA 0.06111381    FALSE            TRUE FALSE
# 54585     gc19_pc.cds::gencode::ATM::ENSG00000149311.13 9.003907e-05      NA 0.06478394    FALSE            TRUE FALSE
# 57623    gc19_pc.cds::gencode::DDX3X::ENSG00000215301.5 1.034343e-04      NA 0.07194147    FALSE            TRUE FALSE
# 61715   gc19_pc.cds::gencode::KMT2C::ENSG00000055609.13 1.145728e-04      NA 0.07689910    FALSE            TRUE FALSE
# 65067 gc19_pc.cds::gencode::PDE4DIP::ENSG00000178104.15 1.464351e-04      NA 0.08947021    FALSE            TRUE FALSE
# 69511  gc19_pc.cds::gencode::TGFBR2::ENSG00000163513.13 3.497148e-04      NA 0.16114422    FALSE            TRUE FALSE
# 67267  gc19_pc.cds::gencode::RPS6KA3::ENSG00000177189.8 3.958300e-04      NA 0.17626731    FALSE           FALSE FALSE

ann_PCAWG_ID_complement[which(ann_PCAWG_ID_complement$PCAWG_IDs %in% TL_exclusive_calls),]
# PCAWG_IDs length nMut nSample    N type_of_element in_CGC in_CGC_literature in_CGC_new in_oncoKB in_pcawg
# 1:     gc19_pc.cds::gencode::ATM::ENSG00000149311.13   9171   71      61 2253     gc19_pc.cds   TRUE              TRUE       TRUE      TRUE     TRUE
# 2:    gc19_pc.cds::gencode::DDX3X::ENSG00000215301.5   2165   31      28 2253     gc19_pc.cds   TRUE              TRUE       TRUE      TRUE     TRUE
# 3:    gc19_pc.cds::gencode::HRAS::ENSG00000174775.12    633   12      11 2253     gc19_pc.cds   TRUE              TRUE       TRUE      TRUE     TRUE
# 4:   gc19_pc.cds::gencode::KMT2C::ENSG00000055609.13  14871  107     101 2253     gc19_pc.cds   TRUE              TRUE       TRUE      TRUE     TRUE
# 5: gc19_pc.cds::gencode::PDE4DIP::ENSG00000178104.15   9154   91      71 2253     gc19_pc.cds   TRUE              TRUE       TRUE      TRUE    FALSE
# 6:   gc19_pc.cds::gencode::RBM10::ENSG00000182872.11   2788   30      29 2253     gc19_pc.cds   TRUE              TRUE       TRUE      TRUE     TRUE
# 7:    gc19_pc.cds::gencode::RNF43::ENSG00000108375.8   2654   28      28 2253     gc19_pc.cds   TRUE              TRUE       TRUE      TRUE     TRUE
# 8:  gc19_pc.cds::gencode::RPS6KA3::ENSG00000177189.8   2193   25      24 2253     gc19_pc.cds  FALSE              TRUE      FALSE     FALSE     TRUE
# 9:  gc19_pc.cds::gencode::TGFBR2::ENSG00000163513.13   1779   24      21 2253     gc19_pc.cds   TRUE              TRUE       TRUE      TRUE     TRUE


TL_hits_nonTRUE <- TL_cds[which(!TL_cds$is_TP),]
TL_hits_nonTRUE <- ann_PCAWG_ID_complement[which(ann_PCAWG_ID_complement$PCAWG_IDs %in% TL_hits_nonTRUE$PCAWG_IDs),]
# PCAWG_IDs length nMut nSample    N type_of_element in_CGC in_CGC_literature in_CGC_new in_oncoKB in_pcawg
# 1:    gc19_pc.cds::gencode::AC026310.1::ENSG00000268865.1    741   17      17 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE    FALSE
# 2:        gc19_pc.cds::gencode::CHRNB2::ENSG00000160716.4   1509   25      25 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE     TRUE
# 3: gc19_pc.cds::gencode::CTD-2144E22.5::ENSG00000179755.2    522   27      27 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE    FALSE
# 4:         gc19_pc.cds::gencode::EPHA2::ENSG00000142627.9   2931   32      32 2253     gc19_pc.cds  FALSE              TRUE      FALSE     FALSE     TRUE
# 5:           gc19_pc.cds::gencode::EYS::ENSG00000188107.9   9646  142     125 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE    FALSE
# 6:          gc19_pc.cds::gencode::FLG::ENSG00000143631.10  11965  154     132 2253     gc19_pc.cds  FALSE              TRUE      FALSE     FALSE    FALSE
# 7:         gc19_pc.cds::gencode::H3F3A::ENSG00000163041.5    501   12      12 2253     gc19_pc.cds   TRUE              TRUE       TRUE     FALSE     TRUE
# 8:      gc19_pc.cds::gencode::HIST1H4D::ENSG00000188987.2    312   23      11 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE     TRUE
# 9:        gc19_pc.cds::gencode::KANSL1::ENSG00000120071.8   3318   34      33 2253     gc19_pc.cds  FALSE              TRUE      FALSE     FALSE     TRUE
# 10:         gc19_pc.cds::gencode::LRP12::ENSG00000147650.7   2580   37      32 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE    FALSE
# 11:        gc19_pc.cds::gencode::OR10G8::ENSG00000234560.3    936   26      26 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE    FALSE
# 12:        gc19_pc.cds::gencode::OR2T34::ENSG00000183310.2    857   18      18 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE    FALSE
# 13:         gc19_pc.cds::gencode::OR5F1::ENSG00000149133.1    945   32      31 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE    FALSE
# 14:         gc19_pc.cds::gencode::OR5W2::ENSG00000187612.1    933   39      34 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE    FALSE
# 15:        gc19_pc.cds::gencode::OR7A10::ENSG00000127515.1    930   20      20 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE     TRUE
# 16:        gc19_pc.cds::gencode::POTEE::ENSG00000188219.10   2674   28      28 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE    FALSE
# 17:         gc19_pc.cds::gencode::POTEJ::ENSG00000222038.3   2319   19      18 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE    FALSE
# 18:        gc19_pc.cds::gencode::PROS1::ENSG00000184500.10   2127   29      29 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE    FALSE
# 19:       gc19_pc.cds::gencode::RPS6KA3::ENSG00000177189.8   2193   25      24 2253     gc19_pc.cds  FALSE              TRUE      FALSE     FALSE     TRUE
# 20:         gc19_pc.cds::gencode::TOR4A::ENSG00000198113.2   1272   19      19 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE    FALSE
# 21:       gc19_pc.cds::gencode::TRIM49C::ENSG00000204449.2   1196   18      18 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE    FALSE

TL_hits_nonTRUE[which(TL_hits_nonTRUE$in_pcawg |
                        TL_hits_nonTRUE$in_oncoKB |
                        TL_hits_nonTRUE$in_CGC_new |
                        TL_hits_nonTRUE$in_CGC_literature |
                        TL_hits_nonTRUE$in_CGC),]

# PCAWG_IDs length nMut nSample    N type_of_element in_CGC in_CGC_literature in_CGC_new in_oncoKB in_pcawg
# 1:   gc19_pc.cds::gencode::CHRNB2::ENSG00000160716.4   1509   25      25 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE     TRUE
# 2:    gc19_pc.cds::gencode::EPHA2::ENSG00000142627.9   2931   32      32 2253     gc19_pc.cds  FALSE              TRUE      FALSE     FALSE     TRUE
# 3:     gc19_pc.cds::gencode::FLG::ENSG00000143631.10  11965  154     132 2253     gc19_pc.cds  FALSE              TRUE      FALSE     FALSE    FALSE
# 4:    gc19_pc.cds::gencode::H3F3A::ENSG00000163041.5    501   12      12 2253     gc19_pc.cds   TRUE              TRUE       TRUE     FALSE     TRUE
# 5: gc19_pc.cds::gencode::HIST1H4D::ENSG00000188987.2    312   23      11 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE     TRUE
# 6:   gc19_pc.cds::gencode::KANSL1::ENSG00000120071.8   3318   34      33 2253     gc19_pc.cds  FALSE              TRUE      FALSE     FALSE     TRUE
# 7:   gc19_pc.cds::gencode::OR7A10::ENSG00000127515.1    930   20      20 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE     TRUE
# 8:  gc19_pc.cds::gencode::RPS6KA3::ENSG00000177189.8   2193   25      24 2253     gc19_pc.cds  FALSE              TRUE      FALSE     FALSE     TRUE
# tissues
# 1:                                                    meta_Carcinoma,Pancan-no-skin-melanoma-lymph
# 2:           meta_Adenocarcinoma,meta_Carcinoma,meta_Digestive_tract,Pancan-no-skin-melanoma-lymph
# 3:                                                                                            <NA>
#   4:                                                      Pancan-no-skin-melanoma-lymph,meta_Sarcoma
# 5:                                              meta_Digestive_tract,Pancan-no-skin-melanoma-lymph
# 6:                                      meta_Carcinoma,Pancan-no-skin-melanoma-lymph,Skin-Melanoma
# 7:                                                                            meta_Digestive_tract
# 8: meta_Adenocarcinoma,meta_Carcinoma,meta_Digestive_tract,Liver-HCC,Pancan-no-skin-melanoma-lymph


TL_hits_nonTRUE[which(!(TL_hits_nonTRUE$in_pcawg |
                        TL_hits_nonTRUE$in_oncoKB |
                        TL_hits_nonTRUE$in_CGC_new |
                        TL_hits_nonTRUE$in_CGC_literature |
                        TL_hits_nonTRUE$in_CGC)),]


# PCAWG_IDs length nMut nSample    N type_of_element in_CGC in_CGC_literature in_CGC_new in_oncoKB in_pcawg tissues
# 1:    gc19_pc.cds::gencode::AC026310.1::ENSG00000268865.1    741   17      17 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE    FALSE    <NA>
#   2: gc19_pc.cds::gencode::CTD-2144E22.5::ENSG00000179755.2    522   27      27 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE    FALSE    <NA>
#   3:           gc19_pc.cds::gencode::EYS::ENSG00000188107.9   9646  142     125 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE    FALSE    <NA>
#   4:         gc19_pc.cds::gencode::LRP12::ENSG00000147650.7   2580   37      32 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE    FALSE    <NA>
#   5:        gc19_pc.cds::gencode::OR10G8::ENSG00000234560.3    936   26      26 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE    FALSE    <NA>
#   6:        gc19_pc.cds::gencode::OR2T34::ENSG00000183310.2    857   18      18 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE    FALSE    <NA>
#   7:         gc19_pc.cds::gencode::OR5F1::ENSG00000149133.1    945   32      31 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE    FALSE    <NA>
#   8:         gc19_pc.cds::gencode::OR5W2::ENSG00000187612.1    933   39      34 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE    FALSE    <NA>
#   9:        gc19_pc.cds::gencode::POTEE::ENSG00000188219.10   2674   28      28 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE    FALSE    <NA>
#   10:         gc19_pc.cds::gencode::POTEJ::ENSG00000222038.3   2319   19      18 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE    FALSE    <NA>
#   11:        gc19_pc.cds::gencode::PROS1::ENSG00000184500.10   2127   29      29 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE    FALSE    <NA>
#   12:         gc19_pc.cds::gencode::TOR4A::ENSG00000198113.2   1272   19      19 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE    FALSE    <NA>
#   13:       gc19_pc.cds::gencode::TRIM49C::ENSG00000204449.2   1196   18      18 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE    FALSE    <NA>







gbm_exclusive_calls <- gbm_cds$PCAWG_IDs[which(!gbm_cds$PCAWG_IDs %in% TL_cds$PCAWG_IDs)]
gbm_exclusive_calls <- ann_PCAWG_ID_complement[which(ann_PCAWG_ID_complement$PCAWG_IDs %in% gbm_exclusive_calls),]

# PCAWG_IDs length nMut nSample    N type_of_element in_CGC in_CGC_literature in_CGC_new in_oncoKB in_pcawg         tissues
# 1: gc19_pc.cds::gencode::ADAMTSL3::ENSG00000156218.8   5112   56      54 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE    FALSE            <NA>
#   2:     gc19_pc.cds::gencode::GRM3::ENSG00000198822.6   2781   60      55 2253     gc19_pc.cds  FALSE             FALSE       TRUE      TRUE    FALSE            <NA>
#   3:      gc19_pc.cds::gencode::IVL::ENSG00000163207.5   1700   37      34 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE     TRUE Stomach-AdenoCa
# 4:    gc19_pc.cds::gencode::KCNV1::ENSG00000164794.4   1503   35      32 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE    FALSE            <NA>
#   5:   gc19_pc.cds::gencode::KIF1A::ENSG00000130294.10   5532   62      57 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE    FALSE            <NA>
#   6:    gc19_pc.cds::gencode::MUC17::ENSG00000169876.9  13482  123     113 2253     gc19_pc.cds  FALSE              TRUE      FALSE     FALSE    FALSE            <NA>
#   7:    gc19_pc.cds::gencode::PCSK2::ENSG00000125851.5   1917   33      33 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE    FALSE            <NA>
#   8:   gc19_pc.cds::gencode::RADIL::ENSG00000157927.12   3228   50      45 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE    FALSE            <NA>
#   9:     gc19_pc.cds::gencode::RTL1::ENSG00000254656.1   4077   59      59 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE    FALSE            <NA>
#   10:     gc19_pc.cds::gencode::RYR1::ENSG00000196218.7  15096  136     129 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE    FALSE            <NA>
#   11:    gc19_pc.cds::gencode::SCN4A::ENSG00000007314.7   5515   74      53 2253     gc19_pc.cds  FALSE             FALSE      FALSE     FALSE    FALSE            <NA>


#missed by TL:
# > all_TL_cds[which(all_TL_cds$PCAWG_IDs == 'gc19_pc.cds::gencode::GRM3::ENSG00000198822.6'),]
# PCAWG_IDs      p_value q_value      fdr positive is_goldStandard is_TP
# 60098 gc19_pc.cds::gencode::GRM3::ENSG00000198822.6 0.0004374237      NA 0.182487    FALSE            TRUE FALSE
# > all_gbm_cds[which(all_gbm_cds$PCAWG_IDs == 'gc19_pc.cds::gencode::GRM3::ENSG00000198822.6'),]
# PCAWG_IDs      p_value q_value        fdr positive is_goldStandard is_TP
# 60098 gc19_pc.cds::gencode::GRM3::ENSG00000198822.6 7.411766e-06      NA 0.00923808     TRUE            TRUE  TRUE
########################################################################################################
########################################################################################################
########################################################################################################
all_gbm_nc <- comp_pRes[["GBM"]][["non_coding"]]
gbm_nc <- all_gbm_nc[which(all_gbm_nc$fdr < .05),]

all_eMET_nc <- comp_pRes[['eMET']][['non_coding']]
eMET_nc <- all_eMET_nc[which(all_eMET_nc$fdr < .05),]
eMET_nc <- left_join(eMET_nc, ann, by = 'PCAWG_IDs')
eMET_table_nc <- eMET_nc[, c("PCAWG_IDs", "p_value", "fdr", "is_TP", "type_of_element", "in_CGC",
                         "in_CGC_literature", "in_CGC_new", "in_oncoKB", "in_pcawg")]
fwrite(eMET_table_nc, file = '../../make_features/external/BMR/tables/nc_sigHits_eMET.tsv',
       sep = '\t')




which(!eMET_nc$PCAWG_IDs %in% gbm_nc$PCAWG_IDs)

eMET_nc_nonTRUE <- eMET_nc[which(!eMET_nc$is_TP),]
eMET_nc_nonTRUE <- ann_PCAWG_ID_complement[which(ann_PCAWG_ID_complement$PCAWG_IDs %in% eMET_nc_nonTRUE$PCAWG_IDs),]

eMET_nc_nonTRUE[which(!(eMET_nc_nonTRUE$in_pcawg |
                          eMET_nc_nonTRUE$in_oncoKB |
                          eMET_nc_nonTRUE$in_CGC_new |
                          eMET_nc_nonTRUE$in_CGC_literature |
                          eMET_nc_nonTRUE$in_CGC)),]



# 1:       gc19_pc.3utr::gencode::ADH1B::ENSG00000196616.8   2825   32      32 2253     gc19_pc.3utr  FALSE
# 2:       gc19_pc.3utr::gencode::BHMT::ENSG00000145692.10   1169   20      20 2253     gc19_pc.3utr  FALSE
# 3:      gc19_pc.3utr::gencode::BRINP2::ENSG00000198797.6    894   25      22 2253     gc19_pc.3utr  FALSE
# 4:      gc19_pc.3utr::gencode::GLYR1::ENSG00000140632.12   2203   29      22 2253     gc19_pc.3utr  FALSE
# 5:      gc19_pc.3utr::gencode::IFI16::ENSG00000163565.14    254   10      10 2253     gc19_pc.3utr  FALSE
# 6:        gc19_pc.3utr::gencode::PDPR::ENSG00000090857.9   4421   36      35 2253     gc19_pc.3utr  FALSE
# 7:       gc19_pc.3utr::gencode::TCP10::ENSG00000203690.7   4372   42      40 2253     gc19_pc.3utr  FALSE
# 8: gc19_pc.promCore::gencode::COL4A2::ENSG00000134871.13    794   17      17 2253 gc19_pc.promCore  FALSE
# 9:   gc19_pc.promCore::gencode::OR5T3::ENSG00000172489.5    200   11      11 2253 gc19_pc.promCore  FALSE
# 10:   gc19_pc.promCore::gencode::SRPRB::ENSG00000144867.7    774   16      16 2253 gc19_pc.promCore  FALSE
# 11:   gc19_pc.promCore::gencode::TPTE::ENSG00000166157.12    432   28      26 2253 gc19_pc.promCore  FALSE
# 12: gc19_pc.promCore::gencode::ZSCAN5B::ENSG00000197213.5    675   23      23 2253 gc19_pc.promCore  FALSE
# in_CGC_literature in_CGC_new in_oncoKB in_pcawg tissues
# 1:             FALSE      FALSE     FALSE    FALSE    <NA>
#   2:             FALSE      FALSE     FALSE    FALSE    <NA>
#   3:             FALSE      FALSE     FALSE    FALSE    <NA>
#   4:             FALSE      FALSE     FALSE    FALSE    <NA>
#   5:             FALSE      FALSE     FALSE    FALSE    <NA>
#   6:             FALSE      FALSE     FALSE    FALSE    <NA>
#   7:             FALSE      FALSE     FALSE    FALSE    <NA>
#   8:             FALSE      FALSE     FALSE    FALSE    <NA>
#   9:             FALSE      FALSE     FALSE    FALSE    <NA>
#   10:             FALSE      FALSE     FALSE    FALSE    <NA>
#   11:             FALSE      FALSE     FALSE    FALSE    <NA>
#   12:             FALSE      FALSE     FALSE    FALSE    <NA>
#   > 

######## gc19_pc.promCore::gencode::TPTE::ENSG00000166157.12 in cncdatabase



eMET_nc[which(!eMET_nc$PCAWG_IDs %in% gbm_nc$PCAWG_IDs),]
# PCAWG_IDs      p_value q_value        fdr positive is_goldStandard is_TP
# 27415    gc19_pc.3utr::gencode::DUSP22::ENSG00000112679.10 1.803524e-05      NA 0.01857979     TRUE            TRUE  TRUE
# 29850     gc19_pc.3utr::gencode::IFI16::ENSG00000163565.14 3.078162e-05      NA 0.02825165     TRUE           FALSE FALSE
# 89667 gc19_pc.promCore::gencode::ZNF595::ENSG00000197701.7 4.304382e-05      NA 0.03720116     TRUE           FALSE FALSE
# 23498      gc19_pc.3utr::gencode::ADH1B::ENSG00000196616.8 4.311191e-05      NA 0.03720116     TRUE           FALSE FALSE
# 78871  gc19_pc.promCore::gencode::HDAC1::ENSG00000116478.7 6.179196e-05      NA 0.04969856     TRUE            TRUE  TRUE
# 82832  gc19_pc.promCore::gencode::OR5T3::ENSG00000172489.5 6.202536e-05      NA 0.04969856     TRUE           FALSE FALSE

ann_PCAWG_ID_complement[which(ann_PCAWG_ID_complement$PCAWG_IDs %in% eMET_nc$PCAWG_IDs[which(!eMET_nc$PCAWG_IDs %in% gbm_nc$PCAWG_IDs)]),]
# PCAWG_IDs length nMut nSample    N  type_of_element in_CGC in_CGC_literature
# 1:      gc19_pc.3utr::gencode::ADH1B::ENSG00000196616.8   2825   32      32 2253     gc19_pc.3utr  FALSE             FALSE
# 2:    gc19_pc.3utr::gencode::DUSP22::ENSG00000112679.10   2796   58      56 2253     gc19_pc.3utr  FALSE             FALSE
# 3:     gc19_pc.3utr::gencode::IFI16::ENSG00000163565.14    254   10      10 2253     gc19_pc.3utr  FALSE             FALSE
# 4:  gc19_pc.promCore::gencode::HDAC1::ENSG00000116478.7    644   13      13 2253 gc19_pc.promCore  FALSE             FALSE
# 5:  gc19_pc.promCore::gencode::OR5T3::ENSG00000172489.5    200   11      11 2253 gc19_pc.promCore  FALSE             FALSE
# 6: gc19_pc.promCore::gencode::ZNF595::ENSG00000197701.7    403   26      25 2253 gc19_pc.promCore  FALSE             FALSE
# in_CGC_new in_oncoKB in_pcawg                                                                      tissues
# 1:      FALSE     FALSE    FALSE                                                                         <NA>
#   2:      FALSE      TRUE    FALSE                                                                         <NA>
#   3:      FALSE     FALSE    FALSE                                                                         <NA>
#   4:      FALSE      TRUE     TRUE          meta_Carcinoma,Head-SCC,Pancan-no-skin-melanoma-lymph,meta_Squamous
# 5:      FALSE     FALSE    FALSE                                                                         <NA>
#   6:      FALSE     FALSE     TRUE meta_Adenocarcinoma,CNS-Medullo,meta_Carcinoma,Pancan-no-skin-melanoma-lymph

#############################################
gbm_nc[which(!gbm_nc$PCAWG_IDs %in% eMET_nc$PCAWG_IDs),]

# PCAWG_IDs      p_value q_value        fdr positive is_goldStandard is_TP
# 89689  gc19_pc.promCore::gencode::ZNF627::ENSG00000198551.5 8.941981e-06      NA 0.01062086     TRUE           FALSE FALSE
# 100629      gc19_pc.ss::gencode::ZFAND2B::ENSG00000158552.8 1.941799e-05      NA 0.01921981     TRUE           FALSE FALSE
# 94794        gc19_pc.ss::gencode::KDM6A::ENSG00000147050.10 3.227105e-05      NA 0.02883232     TRUE            TRUE  TRUE

# ann_PCAWG_ID_complement[which(ann_PCAWG_ID_complement$PCAWG_IDs %in% gbm_nc$PCAWG_IDs[which(!gbm_nc$PCAWG_IDs %in% eMET_nc$PCAWG_IDs)])]
# PCAWG_IDs length nMut nSample    N  type_of_element in_CGC in_CGC_literature
# 1: gc19_pc.promCore::gencode::ZNF627::ENSG00000198551.5    848   18      17 2253 gc19_pc.promCore  FALSE             FALSE
# 2:       gc19_pc.ss::gencode::KDM6A::ENSG00000147050.10    779   14      14 2253       gc19_pc.ss   TRUE              TRUE
# 3:      gc19_pc.ss::gencode::ZFAND2B::ENSG00000158552.8    255    9       8 2253       gc19_pc.ss  FALSE             FALSE
# in_CGC_new in_oncoKB in_pcawg tissues
# 1:      FALSE     FALSE    FALSE    <NA>
#   2:       TRUE      TRUE    FALSE    <NA>
#   3:      FALSE     FALSE    FALSE    <NA>


gbm_nc_nonTRUE <- gbm_nc[which(!gbm_nc$is_TP),]
gbm_nc_nonTRUE <- ann_PCAWG_ID_complement[which(ann_PCAWG_ID_complement$PCAWG_IDs %in% gbm_nc_nonTRUE$PCAWG_IDs),]

gbm_nc_nonTRUE[which(!(gbm_nc_nonTRUE$in_pcawg |
                         gbm_nc_nonTRUE$in_oncoKB |
                         gbm_nc_nonTRUE$in_CGC_new |
                         gbm_nc_nonTRUE$in_CGC_literature |
                         gbm_nc_nonTRUE$in_CGC)),]

########################################## Revise compare tools ####################
rm(list = ls())

# setwd('C:/Active/projects/iDriver/iDriver/')

library(data.table)
library(openxlsx)
library(dplyr)
library(pROC)
library(PRROC)

source("benchmark/functions_benchmark.R")
source("benchmark/functions_annotate_elements.R")

path_procPCAWG_res <- "../extdata/procInput/PCAWG_results/"
path_proccessedGE <- "../extdata/procInput/"
# tissue <-  "Pancan-no-skin-melanoma-lymph"
path_save_benchRes <- "../../make_features/external/BMR/output/Res_reviewerComments/driver_bench/"


included_cohorts <-  c("Pancan-no-skin-melanoma-lymph", "Liver-HCC", "ColoRect-AdenoCA" ,
                       "Uterus-AdenoCA" , "Kidney-RCC", "Lung-SCC", "Biliary-AdenoCA",
                       "Stomach-AdenoCA", "Skin-Melanoma", "Panc-Endocrine", "Head-SCC",
                       "Breast-AdenoCa", "Bladder-TCC", "Eso-AdenoCa", 
                       "Lymph-BNHL", "Lymph-CLL",
                       "CNS-GBM", "Panc-AdenoCA" , "Lung-AdenoCA" ,"Prost-AdenoCA",
                       "Ovary-AdenoCA" , "Bone-Leiomyo", "CNS-Medullo","Bone-Osteosarc")



sig_definition_methods <- c("fdr")
sig_definition_thresholds <- c(.05)

add_newMethod_to_pRes <- function(pRes, path_newRes){
  
  newMethod <- data.frame(fread(path_newRes))
  newMethod <- newMethod[which(!grepl("lncrna.ncrna", newMethod$binID)),]
  newMethod <- newMethod[which(!grepl("lncrna.promCore", newMethod$binID)),]
  # newMethod <- newMethod[which(newMethod$nSample > 1),]
  idx_pVal <- which(colnames(newMethod) %in% c("pValues", "p_value",
                                               "raw_p", "pvals", "pp_element",
                                               "raw_p_binom", "PVAL_MUT_BURDEN" )) 
  idx_ID <- which(colnames(newMethod) %in% c("binID", "PCAWG_ID", 'id', 'ELT'))
  
  newMethod <- data.frame("PCAWG_IDs" = newMethod[,idx_ID], 
                          "p_value" =  newMethod[,idx_pVal],
                          "q_value" = rep(NA, nrow(newMethod)),
                          "fdr" =  p.adjust(newMethod[,idx_pVal], "fdr"))
  
  newMethod <- newMethod[which(!is.na(newMethod$p_value)),]
  
  pRes$newMethod <- newMethod
  class(pRes) = "pRes"
  pRes
}



save_Measures_inSingletable <- function(path_save, newRESULTS, PATHs_newResults, pRes,
                                        sig_definition_methods,
                                        sig_definition_thresholds,
                                        ann_PCAWG_ID_complement, tissue, based_on,
                                        compareRank_new = list(FALSE, NA)){
  
  if (length(PATHs_newResults) != 0){
    for (i in 1:length(PATHs_newResults)) {
      print(PATHs_newResults[i])
      pRes <- add_newMethod_to_pRes(pRes, PATHs_newResults[i])
      names(pRes) <- c(names(pRes)[1:(length(pRes) -1)], newRESULTS[i])
    }
  }
  
  pRes <- pRes[which(names(pRes) %in% newRESULTS)]
  
  
  
  for (j in 1:length(sig_definition_methods)) {
    sigCriteria <- define_significant_criteria(sig_definition_methods[j],
                                               sig_definition_thresholds[j])
    
    st1 <- computeStat_1(pRes, sigCriteria, ann_PCAWG_ID_complement, based_on)
    st_2 <- computeStat_2(pRes, sigCriteria, ann_PCAWG_ID_complement, based_on)$tables
    
    CDS_tab1 <- cbind(st1$CDS, "method" = rownames(st1$CDS))
    CDS <- full_join(CDS_tab1, st_2$CDS)
    
    NC_tab1 <- cbind(st1$non_coding, "method" = rownames(st1$non_coding))
    non_coding <- full_join(NC_tab1, st_2$non_coding)
    
    tab_all <- full_join(CDS, non_coding, by = "method")
    
    colnames(CDS) <- gsub("CDS_", "", colnames(CDS))
    colnames(non_coding) <- gsub("NC_", "", colnames(non_coding))
    
    
    
    Measures <- list("CDS" = CDS, "non_coding" = non_coding,
                     "all" = tab_all)
    
    df <- tab_all
    col_order <- c("method", "CDS_nTPs", "CDS_nHits",
                   "CDS_PRECISIONs", "CDS_Recalls", "CDS_F1", "CDS_AUC",
                   "CDS_AUPR", "CDS_n_elemnts", "nc_n_elemnts", "NC_nTPs",
                   "NC_nHits", "NC_PRECISIONs", "NC_Recalls", "NC_F1",
                   "NC_AUC", "NC_AUPR")
    
    if (sig_definition_methods[j] == "fdr" & compareRank_new[[1]]) {
      compare_nTPs <- c()
      for(k in 1:nrow(df)){
        sigCr_cds <- define_significant_criteria("fixedNumberOfElems",
                                                 df$CDS_nHits[k] )
        st1_nTP_cds <- computeStat_1(pRes, sigCr_cds, ann_PCAWG_ID_complement, based_on)
        sigCr_nc <- define_significant_criteria("fixedNumberOfElems",
                                                df$NC_nHits[k] )
        st1_nTP_nc <- computeStat_1(pRes, sigCr_nc, ann_PCAWG_ID_complement, based_on)
        
        compare_nTPs <- rbind(compare_nTPs,
                              c(st1_nTP_cds$CDS[compareRank_new[[2]], "CDS_nTPs"],
                                st1_nTP_nc$non_coding[compareRank_new[[2]], "NC_nTPs"]))
        
      }
      colnames(compare_nTPs) <- c("CDS_nTPcompare", "NC_nTPcompare")
      sigCr_cds2 <- define_significant_criteria("fixedNumberOfElems",
                                                df[which(df$method==compareRank_new[[2]]),
                                                   which(colnames(df)=="CDS_nHits")])
      st_nTP_cds <- computeStat_1(pRes, sigCr_cds2, ann_PCAWG_ID_complement, based_on)
      CDS_nTPcompare_reverse=st_nTP_cds$CDS$CDS_nTPs
      df <- cbind(df, CDS_nTPcompare_reverse, compare_nTPs)
      col_order <- c("method", "CDS_nTPs", "CDS_nHits", "CDS_nTPcompare",
                     "CDS_nTPcompare_reverse",
                     "CDS_PRECISIONs", "CDS_Recalls", "CDS_F1", "CDS_AUC",
                     "CDS_AUPR", "CDS_n_elemnts", "nc_n_elemnts", "NC_nTPs",
                     "NC_nHits", "NC_nTPcompare", "NC_PRECISIONs", "NC_Recalls",
                     "NC_F1", "NC_AUC", "NC_AUPR")
    }
    
    # Use the select function to reorder the columns
    df_reordered <- df %>% select(col_order)
    
    dir.create(paste0(path_save, tissue, "/tables/"),
               showWarnings = F,
               recursive = T)
    write.csv(df_reordered, file = paste0(path_save, tissue,
                                          "/tables/table_GoldStd_basedon_",
                                          based_on,
                                          sigCriteria$method, ".csv"))
    
    save(Measures, file = paste0(path_save, tissue,
                                 "/tables/Measures_GoldStd_basedon_", based_on,
                                 sigCriteria$method, ".RData"))
  }
  
}




for (tissue in included_cohorts) {
  print(tissue)
  
  ########### 1) load the pRes object and annotated PCAWG_IDs ###########
  ann_PCAWG_ID_complement <- fread(paste0(path_proccessedGE, "ann_PCAWG_ID_complement.csv"))
  load(paste0(path_procPCAWG_res, tissue, ".RData"))
  
  
  ########### 2) add new result(s) to pRes for computing measure stats and saving the measurment results ##########
  newRESULTS <- c("eMET", 'DP', 'Dig', 'ActivedriverWGS')
  
  PATHs_newResults <- c(paste0("../../make_features/external/BMR/output/reviewerComments/", tissue, "/eMET/inference/eMET_inference.tsv"),
                        paste0("../../DriverPower/output/", tissue, "/final_DP_result.tsv"),
                        paste0("../../Dig/output/elemDriver/", tissue, "_final_Dig_result.tsv"),
                        paste0("../../ActiveDriverWGSR/output/", tissue,"/final_AD_result.tsv"))
                        
  
  
  ## use PCAWG drivers/ CGC as gold standard for defining true positives
  
  driver_based_on <- c("in_CGC_new", "all", "in_pcawg", "in_oncoKB", "any")
  
  for (based_on in driver_based_on) {
    save_Measures_inSingletable(path_save_benchRes, newRESULTS, PATHs_newResults, pRes, 
                                sig_definition_methods,
                                sig_definition_thresholds,
                                ann_PCAWG_ID_complement, tissue, based_on #,
                                # compareRank_new = list(TRUE, "GBM")
    )
    
    
  }
  
}



################################################################################
rm(list = ls())
included_cohorts <- c("Pancan-no-skin-melanoma-lymph", "Liver-HCC", "ColoRect-AdenoCA" ,
                      "Uterus-AdenoCA" , "Kidney-RCC", "Lung-SCC", "Biliary-AdenoCA",
                      "Stomach-AdenoCA", "Skin-Melanoma", "Panc-Endocrine", "Head-SCC",
                      "Breast-AdenoCa", "Bladder-TCC", "Eso-AdenoCa",
                      "Lymph-BNHL", "Lymph-CLL",
                      "CNS-GBM", "Panc-AdenoCA" , "Lung-AdenoCA" ,"Prost-AdenoCA",
                      "Ovary-AdenoCA" , "Bone-Leiomyo", "CNS-Medullo","Bone-Osteosarc")

files <- paste0('../external/BMR/output/reviewerComments/', included_cohorts, '/eMET/inference/eMET_inference.tsv')

ass <- lapply(files, fread)

# Add a cohort column to each assessment
ass <- Map(function(dt, cohort) {
  dt[, cohort := cohort]
  return(dt)
}, ass, included_cohorts)

drivers <- lapply(ass, function(s){
  s = s[which(s$nMut !=0), ]
  print(nrow(s))
  s$fdr = p.adjust(s$p_value, method = 'fdr')
  s = s[which(s$fdr < 0.05), c('binID', 'nMut', 'nSample', 'length', 'p_value', 'fdr','cohort')]
})

drivers <- do.call(rbind, drivers)

dir.create('../external/BMR/output/Res_reviewerComments/driverDiscovery/', recursive = T, showWarnings = F)
fwrite(drivers, file = '../external/BMR/output/Res_reviewerComments/driverDiscovery/suppTab_eMET_allCohorts.tsv', sep = '\t')


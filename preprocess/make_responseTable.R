rm(list = ls())
library(data.table)
library(rtracklayer)
library(dplyr)

###############################
map_muts <- function(gr, path_bed){

  testGE_gr <- import.bed(path_bed)

  names(testGE_gr) = testGE_gr$name

  GenomicElement <- unlist(lapply(strsplit(names(testGE_gr), "[::]"), function(x){x[1]}))

  lenElement <- width(testGE_gr)

  mcols(testGE_gr) <- DataFrame(mcols(testGE_gr), GenomicElement, lenElement)

  callable_GE <- split(testGE_gr, names(testGE_gr))
  lenElement_new <- sum(width(callable_GE))

  ov <- findOverlaps(gr, callable_GE, ignore.strand = TRUE)

  idx_gr <- queryHits(ov)
  idx_callable_GE <- subjectHits(ov)

  gr_mappedGE <- gr[idx_gr]

  GE_col=unique(mcols(callable_GE, level = "within")[,"GenomicElement"])[idx_callable_GE]
  EL_col = sum(mcols(callable_GE, level = "within")[,"lenElement"])[idx_callable_GE]
  name_col = names(callable_GE)[idx_callable_GE]

  mcols(gr_mappedGE) <- DataFrame(mcols(gr_mappedGE),
                                  GenomicElement=GE_col,
                                  elemenntLength = EL_col,
                                  name = name_col)

  idx_completely_black <- which(lenElement_new == 0)



  if(length(idx_completely_black) != 0){
    lenElement_new <- lenElement_new[-idx_completely_black]
    callable_GE <- callable_GE[-idx_completely_black]
  }


  list(MAF_GE_mapped = gr_mappedGE, all_GEs_grl = callable_GE,
       lenAllTests = lenElement_new)

}



create_response <- function(all_elements, n_donors){

  obs_Muts <- all_elements$MAF_GE_mapped
  df <- as.data.frame(DataFrame(obs_Muts))
  df <- df[,c('name', 'D_id')]


  # resTab <- df %>%
  #   count(name, D_id) %>%
  #   filter(n != 0) %>%
  #   group_by(binID = name) %>%
  #   summarise(nMut = sum(n), nSample = n(), .groups = "drop")

  # Convert df to a data.table (if it's not already one)
  setDT(df)

  # Create the result table using data.table syntax
  resTab <- df[, .N, by = .(name, D_id)][
    N != 0, .(nMut = sum(N), nSample = .N), by = .(binID = name)
  ]

  idx_lenElement <- resTab$binID
  all_lmnt_length <- all_elements$lenAllTests
  ElementLength <- all_lmnt_length[idx_lenElement]
  resTab$length <- ElementLength

  resTab$N <- n_donors

  all_binIds <- names(all_elements[["lenAllTests"]])
  nonMut_ids <- all_binIds[which(!all_binIds %in% resTab$binID)]
  res_nonMutated <- data.frame(cbind('binID' = nonMut_ids,
                                     'nMut' = rep(0, length(nonMut_ids)),
                                     'nSample' = rep(0, length(nonMut_ids)),
                                     'length' = all_lmnt_length[!names(all_lmnt_length) %in% resTab$binID],
                                     'N' = rep(n_donors, length(nonMut_ids))))

  resTab <- rbind(resTab, res_nonMutated)

  return(resTab)
}



save_responseTable <- function(gr, path_bed, path_out, save_name){

  all_elements <- map_muts(gr, path_bed)

  print("************************ mutations mapped to the genomic intervals************************")
  n_donors <- length(unique(gr$D_id))
  resTab <- create_response(all_elements, n_donors)
  print("************************ response table was generated ************************")
  dir.create(paste0(path_out, "/"), showWarnings = FALSE, recursive = TRUE)

  fwrite(resTab, paste0(path_out, "/", save_name,
                        ".tsv"),
         sep = "\t", row.names = F)
  print("************************file saved************************")

}


################################################################################
path_to_gr='../../iDriver/extdata/procInput/mut_PCAWG/pan_cancer_FILTEREDsamples.RData'
load(path_to_gr)
gr <- gr[!duplicated(gr)]
################################################################################
path_beds <- c('../external/database/bins/proccessed/PCAWG_callable.bed6',
                 '../external/database/bins/proccessed/callable_intergenic_intervals_wo_pcawg.bed6',
                 '../external/database/bins/proccessed/intergenic_fixed1M.bed6',
               '../external/database/bins/proccessed/intergenic_fixed100k.bed6',
                 '../external/database/bins/proccessed/intergenic_fixed50k.bed6',
                 '../external/database/bins/proccessed/intergenic_fixed10k.bed6')

path_out <- '../external/BMR/rawInput/responseTabs/'
save_names = c('reg_elems', 'var_bins', '1M_bins', '100k_bins', '50k_bins', '10k_bins')

for(i in 1:length(path_beds)){
  save_responseTable(gr, path_beds[i], path_out, save_names[i])
}

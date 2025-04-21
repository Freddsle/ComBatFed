library(matrixStats)
library(tidyverse)
source("https://raw.githubusercontent.com/andy1764/Distributed-ComBat/main/neuroCombat.R")
source("https://raw.githubusercontent.com/andy1764/Distributed-ComBat/main/neuroCombat_helpers.R")
source("https://raw.githubusercontent.com/andy1764/Distributed-ComBat/main/distributedCombat.R")

rm(distributedCombat_site)

distributedCombat_site <- function(dat, 
                                   batch, 
                                   mod=NULL,
                                   ref.batch=NULL,
                                   central.out=NULL,
                                   eb=TRUE, 
                                   parametric=TRUE,
                                   mean.only=FALSE,
                                   verbose=TRUE, 
                                   file=NULL
){
  if (!is.character(file)) {
    warning("Must specify filename to output results as a file. Currently
            saving output to current workspace only.")
  }
  
  if (is.character(central.out)) {
    load(central.out)
    central.out <- central_out
  }
  
  dat <- as.matrix(dat)
  .checkConstantRows(dat)
  .checkNARows(dat)
  ## Check for missing values
  hasNAs <- any(is.na(dat))
  if (hasNAs & verbose){
    cat(paste0("[neuroCombat] Found ", sum(is.na(dat)), " missing data values. \n"))
  }
  if(mean.only){
    if (verbose) cat("[neuroCombat] Performing ComBat with mean only\n")
  }
  
  ##################### Getting design ############################
  dataDict <- getDataDictDC(batch, mod, verbose=verbose, mean.only=mean.only, ref.batch=ref.batch)
  design <- dataDict[["design"]]
  ####################################################################
  
  
  ############### Site matrices for standardization #################
  # W^T W used in LS estimation
  ls_site <- NULL
  ls_site[[1]] <- crossprod(design)
  ls_site[[2]] <- tcrossprod(t(design), dat)
  
  dataDictOut <- dataDict
  dataDictOut$design <- NULL
  
  # new dataDict with batches within current site
  inclBat <- dataDict$n.batches > 0
  dataDictSite <- dataDict
  dataDictSite$batch <- droplevels(dataDict$batch)
  dataDictSite$batches <- dataDict$batches[inclBat]
  dataDictSite$n.batch <- sum(inclBat)
  dataDictSite$n.batches <- dataDict$n.batches[inclBat]
  dataDictSite$batch.design <- as.matrix(dataDict$batch.design[,inclBat])
  
  # remove reference batch information if reference batch is not in site
  if (!is.null(ref.batch)) {
    if (dataDictSite$ref %in% dataDictSite$batch) {
      dataDictSite$ref <- which(levels(as.factor(dataDictSite$batch))==ref.batch)
    } else {
      dataDictSite$ref <- NULL
      dataDictSite$ref.batch <- NULL
    }
  }
  
  if (is.null(central.out)) {
    site_out <- list(
      ls.site = ls_site,
      dataDict = dataDict,
      sigma.site = NULL
    )
    if (is.character(file)) {
      save(site_out, file = file)
      return(invisible())
    } else {
      return(site_out)
    }
  } 
  
  # If beta.estimates given, get summary statistics for sigma estimation
  
  if (is.null(central.out$var.pooled)) {
    sigma_site <- getSigmaSummary(dat, dataDict, design, hasNAs, central.out)
    
    site_out <- list(
      ls.site = ls_site,
      dataDict = dataDict,
      sigma.site = sigma_site
    )
    if (is.character(file)) {
      save(site_out, file = file)
      return(invisible())
    } else {
      return(site_out)
    }
  }
  
  stdObjects <- getStandardizedDataDC(dat=dat, 
                                      dataDict=dataDict,
                                      design=design,
                                      hasNAs=hasNAs,
                                      central.out=central.out
  )
  s.data <- stdObjects[["s.data"]]
  ####################################################################
  
  
  
  ##################### Getting L/S estimates #######################
  if (verbose) cat("[distributedCombat] Fitting L/S model and finding priors\n")
  naiveEstimators <- getNaiveEstimators(s.data=s.data,
                                        dataDict=dataDictSite, 
                                        hasNAs=hasNAs,
                                        mean.only=mean.only
  )
  ####################################################################
  
  
  ######################### Getting final estimators ####################
  if (eb){
    if (parametric){
      if (verbose) cat("[distributedCombat] Finding parametric adjustments\n")}else{
        if (verbose) cat("[distributedCombat] Finding non-parametric adjustments\n")
      }
    estimators <- getEbEstimators(naiveEstimators=naiveEstimators, 
                                  s.data=s.data, 
                                  dataDict=dataDictSite,
                                  parametric=parametric,
                                  mean.only=mean.only
    )
  } else {
    estimators <- getNonEbEstimators(naiveEstimators=naiveEstimators, dataDict=dataDict)
  }
  ####################################################################
  
  
  
  ######################### Correct data #############################
  if (verbose) cat("[distributedCombat] Adjusting the Data\n")
  bayesdata <- getCorrectedData(dat=dat,
                                s.data=s.data,
                                dataDict=dataDictSite,
                                estimators=estimators,
                                naiveEstimators=naiveEstimators,
                                stdObjects=stdObjects,
                                eb=eb
  )
  ####################################################################
  
  
  # List of estimates:
  estimates <- list(gamma.hat=naiveEstimators[["gamma.hat"]], 
                    delta.hat=naiveEstimators[["delta.hat"]], 
                    gamma.star=estimators[["gamma.star"]],
                    delta.star=estimators[["delta.star"]], 
                    gamma.bar=estimators[["gamma.bar"]], 
                    t2=estimators[["t2"]], 
                    a.prior=estimators[["a.prior"]], 
                    b.prior=estimators[["b.prior"]], 
                    stand.mean=stdObjects[["stand.mean"]], 
                    mod.mean=stdObjects[["mod.mean"]], 
                    # var.pooled=stdObjects[["var.pooled"]],
                    var.pooled=central.out$var.pooledб
                    beta.hat=stdObjects[["beta.hat"]],
                    mod=mod, 
                    batch=batch, 
                    ref.batch=ref.batch, 
                    eb=eb, 
                    parametric=parametric, 
                    mean.only=mean.only,
                    xtx=crossprod(design),
                    xty=tcrossprod(t(design), dat),
                    sigma=getSigmaSummary(dat, dataDict, design, hasNAs, central.out),
                    B_hat = central.out$B.hat
  )
  
  site_out <- list(dat.combat=bayesdata, estimates=estimates)
  if (is.character(file)) {
    save(site_out, file = file)
    return(invisible())
  } else {
    return(site_out)
  }
}


data_path <- "datasets/Breast_cancer_RNASeq/"
global_row_names <- c()

for(lab in c("GSE129508", "GSE149276", "GSE58135")){
  
  dat <- read.table(paste0(data_path, "before/", lab, "/expr_for_correction.tsv"), 
                    header = TRUE, sep = "\t", row.names = 1)
  metadata <- read.table(paste0(data_path, "before/", lab, "/design.tsv"), 
                         header = TRUE, sep = "\t")
  
  # if rows in data contains only zeros, remove them
  dat <- dat[rowSums(dat) != 0, ]
  
  if (length(global_row_names) == 0){
    global_row_names <- rownames(dat)
  } else {
    global_row_names <- intersect(global_row_names, rownames(dat))
  }
}


#############################################################################3
library(jsonlite)

for(lab in c("GSE129508", "GSE149276", "GSE58135")){
  
  dat <- read.table(paste0(data_path, "before/", lab, "/expr_for_correction.tsv"), 
                    header = TRUE, sep = "\t", row.names = 1)
  metadata <- read.table(paste0(data_path, "before/", lab, "/design.tsv"), 
                         header = TRUE, sep = "\t")
  
  dat <- dat[global_row_names, ]
  mod <- metadata$lum %>% as.matrix()
  rownames(mod) <- metadata$sample_id
  bat <- factor(metadata$batch, levels = c("0", "1", "2"))
  dat <- dat[, metadata$sample_id] %>% as.matrix()
  
  corrected_data_ind <- distributedCombat_site(dat, bat, mod,
                                               central.out = "evaluation/d_combat/intermediate/central_step2.Rdata"
  )
  
  data_checks <- list()
  
  data_checks$xtx <- corrected_data_ind$estimates$xtx
  data_checks$xty <- corrected_data_ind$estimates$xty %>% as.data.frame()
  data_checks$corrected_data <- corrected_data_ind$dat.combat %>% as.data.frame()
  data_checks$gamma_hat <- corrected_data_ind$estimates$gamma.hat  %>% as.data.frame()
  data_checks$gamma_bar <- corrected_data_ind$estimates$gamma.bar %>% as.list() %>% .[[1]]
  data_checks$gamma_star <- corrected_data_ind$estimates$gamma.star %>% as.data.frame()
  data_checks$delta_hat <- corrected_data_ind$estimates$delta.hat %>% as.data.frame()
  data_checks$delta_star <- corrected_data_ind$estimates$delta.star %>% as.data.frame()
  data_checks$t2 <- corrected_data_ind$estimates$t2 %>% as.list() %>% .[[1]]
  data_checks$a_prior <- corrected_data_ind$estimates$a.prior %>% as.list() %>% .[[1]]
  data_checks$b_prior <- corrected_data_ind$estimates$b.prior %>% as.list() %>% .[[1]]
  data_checks$stand_mean <- corrected_data_ind$estimates$stand.mean %>% as.data.frame()
  data_checks$mod_mean <- corrected_data_ind$estimates$mod.mean %>% as.data.frame()
  data_checks$pooled_variance <- corrected_data_ind$estimates$var.pooled %>% as.data.frame()
  data_checks$sigma <-corrected_data_ind$estimates$sigma %>% as.data.frame()
  data_checks$B_hat <-corrected_data_ind$estimates$B_hat %>% as.data.frame()
  
  json_data <- toJSON(data_checks, pretty = TRUE, auto_unbox = TRUE,
                      dataframe = "columns")
  write(json_data, file = paste0("evaluation/d_combat/json/", lab, "_D_out.json"))
}

# json_data <- toJSON(my_list, pretty = TRUE, auto_unbox = TRUE)
# write(json_data, file = "output.json")


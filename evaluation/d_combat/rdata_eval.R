library(tidyverse)
# Author: Andrew Chen, andrewac@sas.upenn.edu
# Date: September 7, 2020

#' Distributed ComBat step at each site
#'
#' @param dat A \emph{p x n} matrix (or object coercible by
#'   \link[base]{as.matrix} to a numeric matrix) of observations where \emph{p}
#'   is the number of features and \emph{n} is the number of subjects.
#' @param batch Factor indicating batch. Needs to have the same levels across
#'   all individual sites, but can have multiple batches per site (i.e.
#'   multiple levels in each site)
#' @param mod Optional design matrix of covariates to preserve, usually from 
#'    \link[stats]{model.matrix}. This matrix needs to have the same columns
#'    across sites. The rows must be in the same order as the data columns.
#' @param ref.batch Optional, reference batch used to determine target mean and
#'   variance. Must be specified for all sites.
#' @param central.out Output list from \code{distributedCombat_central}. Output
#'   of \code{distributedCombat_site} will depend on the values of 
#'   \code{central.out}. If \code{NULL}, then the output will be sufficient for
#'   estimation of \code{B.hat}. If \code{B.hat} is provided, then the output
#'   will be sufficient for estimation of \code{sigma} or for harmonization if
#'   \code{mean.only} is \code{TRUE}. If \code{sigma} is provided, then
#'   harmonization will be performed.
#' @param eb If \code{TRUE}, the empirical Bayes step is used to pool
#'   information across features, as per the original ComBat methodology. If
#'   \code{FALSE}, adjustments are made for each feature individually.
#'   Recommended left as \code{TRUE}.
#' @param parametric If \code{TRUE}, parametric priors are used for the
#'   empirical Bayes step, otherwise non-parametric priors are used. See
#'   neuroComBat package for more details. 
#' @param mean.only If \code{TRUE}, distributed ComBat does not harmonize the
#'   variance of features.
#' @param verbose If \code{TRUE}, print progress updates to the console.
#' @param file File name of .Rdata file to export
#' 
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
                    var.pooled=stdObjects[["var.pooled"]],
                    beta.hat=stdObjects[["beta.hat"]],
                    mod=mod, 
                    batch=batch, 
                    ref.batch=ref.batch, 
                    eb=eb, 
                    parametric=parametric, 
                    mean.only=mean.only
  )
  
  site_out <- list(dat.combat=bayesdata, estimates=estimates)
  if (is.character(file)) {
    save(site_out, file = file)
    return(invisible())
  } else {
    return(site_out)
  }
}

#' Distributed ComBat step at analysis core
#' 
#' @param site.outs List or vector of filenames containing site outputs.
#' @param file File name of .Rdata file to export
#' @param ref.batch Optional, reference batch used to determine target mean and
#'   variance
#' @param verbose Whether to print messages to console
distributedCombat_central <- function(site.outs,
                                      file = NULL,
                                      ref.batch = NULL,
                                      verbose = FALSE) {
  if (!is.character(file)) {
    warning("Must specify filename to output results as a file. Currently
            saving output to current workspace only.")
  }
  
  if (is.character(site.outs)) {
    fnames <- site.outs
    site.outs <- lapply(fnames, function(file) {
      load(file)
      site_out
    })
  }
  m <- length(site.outs) # number of sites
  
  # get n.batches and n.array from sites
  batch_levels <- levels(site.outs[[1]]$dataDict$batch)
  n.batches <- Reduce("+", lapply(site.outs, function(x) x$dataDict$n.batches))
  n.batch <- length(n.batches)
  n.array <- sum(n.batches)
  n.arrays <- lapply(site.outs, function(x) x$dataDict$n.array)
  
  # # get reference batch if specified
  if (!is.null(ref.batch)){
    if (!(ref.batch%in%levels(batch))) {
      stop("reference level ref.batch is not found in batch")
    }
    if (verbose){
      cat(paste0("[combat] Using batch=",ref.batch, " as a reference batch \n"))
    }
    ref <- which(batch_levels==ref.batch) # find the reference
  } else {
    ref <- NULL
  }
  
  # check if beta estimates have been given to sites
  step1s <- sapply(site.outs, function(x) is.null(x$sigma.site))
  if (length(unique(step1s)) > 1) {
    stop("Not all sites are at the same step, please confirm with each site.")
  }
  step1 <- all(step1s)
  
  #### Step 1: Get LS estimate across sites ####
  ls1 <- Reduce("+", lapply(site.outs, function(x) x$ls.site[[1]]))
  ls2 <- Reduce("+", lapply(site.outs, function(x) x$ls.site[[2]]))
  B.hat <- crossprod(solve(ls1), ls2)
  
  if (!is.null(ref.batch)) {
    grand.mean <- t(B.hat[ref, ])
  } else {
    grand.mean <- crossprod(n.batches/n.array, B.hat[1:n.batch,])
  }
  stand.mean <- crossprod(grand.mean, t(rep(1,n.array)))
  
  if (step1) {
    central_out <- list(
      B.hat = B.hat,
      stand.mean = stand.mean,
      var.pooled = NULL
    )
    if (is.character(file)) {
      save(central_out, file = file)
      return(invisible())
    } else {
      return(central_out)
    }
  }
  
  # #### Step 2: Get standardization parameters ####
  vars <- lapply(site.outs, function(x) x$sigma.site)
  
  # if ref.batch specified, use estimated variance from reference site
  if (!is.null(ref.batch)){
    var.pooled = vars[[ref]]
  } else {
    var.pooled = rep(0, length(vars[[1]]))
    for (i in 1:m) {
      var.pooled = var.pooled + n.arrays[[i]]*vars[[i]]
    }
    var.pooled = var.pooled/n.array
  }
  
  central_out <- list(
    B.hat = B.hat,
    stand.mean = stand.mean,
    var.pooled = var.pooled
  )
  if (is.character(file)) {
    save(central_out, file = file)
    return(invisible())
  } else {
    return(central_out)
  }
}

# modified to not check design matrix
getDataDictDC <- function(batch, mod, verbose, mean.only, ref.batch=NULL){
  batch <- as.factor(batch)
  n.batch <- nlevels(batch)
  batches <- lapply(levels(batch), function(x)which(batch==x))
  n.batches <- sapply(batches, length)
  n.array  <- sum(n.batches)
  batchmod <- model.matrix(~-1+batch)  
  if (verbose) cat("[combat] Found",nlevels(batch),'batches\n')
  if(any(n.batches==1) & mean.only==FALSE){
    stop("Found one site with only one sample. Consider using the mean.only=TRUE option")
  }
  if (!is.null(ref.batch)){
    if (!(ref.batch%in%levels(batch))) {
      stop("reference level ref.batch is not found in batch")
    }
    if (verbose){
      cat(paste0("[combat] Using batch=",ref.batch, " as a reference batch \n"))
    }
    ref <- which(levels(as.factor(batch))==ref.batch) # find the reference
    batchmod[,ref] <- 1
  } else {
    ref <- NULL
  }
  #combine batch variable and covariates
  design <- cbind(batchmod,mod)
  # check for intercept in covariates, and drop if present
  # check  <- apply(design, 2, function(x) all(x == 1))
  # if(!is.null(ref)){
  #   check[ref] <- FALSE
  # }
  # design <- as.matrix(design[,!check])
  # design <- .checkDesign(design, n.batch)
  n.covariates <- ncol(design)-ncol(batchmod)
  if (verbose) cat("[combat] Adjusting for ",n.covariates,' covariate(s) or covariate level(s)\n')
  out <- list()
  #Making sure to keep track of names:
  names(batches)   <- names(n.batches) <- levels(batch)
  colnames(design) <- gsub("batch", "", colnames(design))
  out[["batch"]] <- batch
  out[["batches"]] <- batches
  out[["n.batch"]] <- n.batch
  out[["n.batches"]] <- n.batches
  out[["n.array"]] <- n.array
  out[["n.covariates"]] <- n.covariates
  out[["design"]] <- design
  out[["batch.design"]] <- design[,1:n.batch]
  out[["ref"]] <- ref
  out[["ref.batch"]] <- ref.batch
  return(out)
}

getSigmaSummary <- function(dat, dataDict, design, hasNAs, central.out){
  batches=dataDict$batches
  n.batches=dataDict$n.batches
  n.array=dataDict$n.array
  n.batch=dataDict$n.batch
  ref.batch=dataDict$ref.batch
  ref=dataDict$ref
  B.hat <- central.out$B.hat
  stand.mean <- central.out$stand.mean[,1:n.array]
  
  if (!hasNAs){
    if (!is.null(ref.batch)){
      ref.dat <- dat[, batches[[ref]]]
      factors <- (n.batches[ref]/(n.batches[ref]-1))
      var.pooled <- rowVars(ref.dat-t(design[batches[[ref]], ]%*%B.hat), na.rm=TRUE)/factors
    } else {
      factors <- (n.array/(n.array-1))
      var.pooled <- rowVars(dat-t(design %*% B.hat), na.rm=TRUE)/factors
    }
  } else {
    if (!is.null(ref.batch)){
      ref.dat <- dat[, batches[[ref]]]  
      ns <- rowSums(!is.na(ref.dat))
      factors <- (ns/(ns-1))
      var.pooled <- rowVars(ref.dat-t(design[batches[[ref]], ]%*%B.hat), na.rm=TRUE)/factors
    } else {
      ns <- rowSums(!is.na(dat))
      factors <- (ns/(ns-1))
      var.pooled <- rowVars(dat-t(design %*% B.hat), na.rm=TRUE)/factors
    }
  }
  
  return(var.pooled)
}

getStandardizedDataDC <- function(dat, dataDict, design, hasNAs, central.out){
  batches=dataDict$batches
  n.batches=dataDict$n.batches
  n.batch=dataDict$n.batch
  n.array=dataDict$n.array
  ref.batch=dataDict$ref.batch
  ref=dataDict$ref
  
  B.hat <- central.out$B.hat
  stand.mean <- central.out$stand.mean[,1:n.array]
  var.pooled <- central.out$var.pooled
  
  if(!is.null(design)){
    tmp <- design
    tmp[,c(1:n.batch)] <- 0
    mod.mean <- t(tmp%*%B.hat)
    #stand.mean <- stand.mean+t(tmp%*%B.hat)
  } else {
    mod.mean <- 0
  }
  s.data <- (dat-stand.mean-mod.mean)/(tcrossprod(sqrt(var.pooled), rep(1,n.array)))
  names(var.pooled) <- rownames(dat)
  rownames(stand.mean) <- rownames(mod.mean) <- rownames(dat)
  colnames(stand.mean) <- colnames(mod.mean) <- colnames(dat)
  return(list(s.data=s.data, 
              stand.mean=stand.mean,
              mod.mean=mod.mean, 
              var.pooled=var.pooled,
              beta.hat=B.hat
  )
  )
}

# Author: Jean-Philippe Fortin, fortin946@gmail.com
# Date: July 14 2020
# Projet repo: github.com/Jfortin1/ComBatHarmonization
# This is a modification of the ComBat function code from the sva package that can be found at
# https://bioconductor.org/packages/release/bioc/html/sva.html 
# The original code is under the Artistic License 2.0.
# The present code is under the MIT license
# If using this code, make sure you agree and accept this license. 


.betaNA <- function(yy,designn){
      designn <- designn[!is.na(yy),]
      yy <- yy[!is.na(yy)]
      B <- solve(crossprod(designn), crossprod(designn, yy))
      B
}

.checkNARows <- function(dat){
	nas <- rowSums(is.na(dat))
	ns <- sum(nas==ncol(dat))
	if (ns>0){
	  message <- paste0(ns, " rows (features) were found to have missing values for all samples. Please remove these rows before running ComBat.")
	  stop(message)
	}
}

.checkConstantRows <- function(dat){
	sds <- rowSds(dat, na.rm=TRUE)
	ns <- sum(sds==0)
	if (ns>0){
	  message <- paste0(ns, " rows (features) were found to be constant across samples. Please remove these rows before running ComBat.")
	  stop(message)
	}
}



.checkDesign <- function(design, n.batch){
  # Check if the design is confounded
  if(qr(design)$rank<ncol(design)){
    if(ncol(design)==(n.batch+1)){
      stop("[combat] The covariate is confounded with batch. Remove the covariate and rerun ComBat.")
    }
    if(ncol(design)>(n.batch+1)){
      if((qr(design[,-c(1:n.batch)])$rank<ncol(design[,-c(1:n.batch)]))){
        stop('The covariates are confounded. Please remove one or more of the covariates so the design is not confounded.')
      } else {
        stop("At least one covariate is confounded with batch. Please remove confounded covariates and rerun ComBat.")
      }
    }
  }
  design
}


getDataDict <- function(batch, mod, verbose, mean.only, ref.batch=NULL){
    batch <- as.factor(batch)
    n.batch <- nlevels(batch)
    batches <- lapply(levels(batch), function(x)which(batch==x))
    n.batches <- sapply(batches, length)
    n.array  <- sum(n.batches)
    batchmod <- model.matrix(~-1+batch)  
    if (verbose) cat("[combat] Found",nlevels(batch),'batches\n')
    if(any(n.batches==1) & mean.only==FALSE){
      stop("Found one site with only one sample. Consider using the mean.only=TRUE option")
    }
    if (!is.null(ref.batch)){
        if (!(ref.batch%in%levels(batch))) {
            stop("reference level ref.batch is not found in batch")
        }
        if (verbose){
          cat(paste0("[combat] Using batch=",ref.batch, " as a reference batch \n"))
        }
        ref <- which(levels(as.factor(batch))==ref.batch) # find the reference
        batchmod[,ref] <- 1
    } else {
        ref <- NULL
    }
    #combine batch variable and covariates
    design <- cbind(batchmod,mod)
    # check for intercept in covariates, and drop if present
    check  <- apply(design, 2, function(x) all(x == 1))
    if(!is.null(ref)){
        check[ref] <- FALSE
    }
    design <- as.matrix(design[,!check])
    design <- .checkDesign(design, n.batch)
    n.covariates <- ncol(design)-ncol(batchmod)
    if (verbose) cat("[combat] Adjusting for ",n.covariates,' covariate(s) or covariate level(s)\n')
      out <- list()
    #Making sure to keep track of names:
    names(batches)   <- names(n.batches) <- levels(batch)
    colnames(design) <- gsub("batch", "", colnames(design))
    out[["batch"]] <- batch
    out[["batches"]] <- batches
    out[["n.batch"]] <- n.batch
    out[["n.batches"]] <- n.batches
    out[["n.array"]] <- n.array
    out[["n.covariates"]] <- n.covariates
    out[["design"]] <- design
    out[["batch.design"]] <- design[,1:n.batch]
    out[["ref"]] <- ref
    out[["ref.batch"]] <- ref.batch
    return(out)
}





getStandardizedData <- function(dat, dataDict, design, hasNAs){
    batches=dataDict$batches
    n.batches=dataDict$n.batches
    n.array=dataDict$n.array
    n.batch=dataDict$n.batch
    ref.batch=dataDict$ref.batch
    ref=dataDict$ref
    .getBetaHat <- function(dat, design, hasNAs){
        if (!hasNAs){
          B.hat <- solve(crossprod(design))
          B.hat <- tcrossprod(B.hat, design)
          B.hat <- tcrossprod(B.hat, dat)
        } else {
          B.hat <- apply(dat, 1, .betaNA, design)
        }
    }
    B.hat <- .getBetaHat(dat=dat, design=design, hasNAs=hasNAs)
    if(!is.null(ref.batch)){
        grand.mean <- t(B.hat[ref, ])
    } else {
        grand.mean <- crossprod(n.batches/n.array, B.hat[1:n.batch,])
    }
    stand.mean <- crossprod(grand.mean, t(rep(1,n.array)))
    if (!hasNAs){
      if (!is.null(ref.batch)){
          ref.dat <- dat[, batches[[ref]]]
          factors <- (n.batches[ref]/(n.batches[ref]-1))
          var.pooled <- rowVars(ref.dat-t(design[batches[[ref]], ]%*%B.hat), na.rm=TRUE)/factors
      } else {
          factors <- (n.array/(n.array-1))
          var.pooled <- rowVars(dat-t(design %*% B.hat), na.rm=TRUE)/factors
      }
    } else {
      if (!is.null(ref.batch)){
          ref.dat <- dat[, batches[[ref]]]  
          ns <- rowSums(!is.na(ref.dat))
          factors <- (ns/(ns-1))
          var.pooled <- rowVars(ref.dat-t(design[batches[[ref]], ]%*%B.hat), na.rm=TRUE)/factors
      } else {
          ns <- rowSums(!is.na(dat))
          factors <- (ns/(ns-1))
          var.pooled <- rowVars(dat-t(design %*% B.hat), na.rm=TRUE)/factors
      }
    }

    if(!is.null(design)){
      tmp <- design
      tmp[,c(1:n.batch)] <- 0
      mod.mean <- t(tmp%*%B.hat)
      #stand.mean <- stand.mean+t(tmp%*%B.hat)
    } else {
      mod.mean <- 0
    }
    s.data <- (dat-stand.mean-mod.mean)/(tcrossprod(sqrt(var.pooled), rep(1,n.array)))
    names(var.pooled) <- rownames(dat)
    rownames(stand.mean) <- rownames(mod.mean) <- rownames(dat)
    colnames(stand.mean) <- colnames(mod.mean) <- colnames(dat)
    return(list(s.data=s.data, 
        stand.mean=stand.mean,
        mod.mean=mod.mean, 
        var.pooled=var.pooled,
        beta.hat=B.hat
        )
    )
}

# Following four find empirical hyper-prior values
aprior <- function(delta.hat){
	m=mean(delta.hat)
	s2=var(delta.hat)
	(2*s2+m^2)/s2
}
bprior <- function(delta.hat){
	m=mean(delta.hat)
	s2=var(delta.hat)
	(m*s2+m^3)/s2
}
apriorMat <- function(delta.hat) {
  m  <- rowMeans2(delta.hat)
  s2 <- rowVars(delta.hat)
  out <- (2*s2+m^2)/s2
  names(out) <- rownames(delta.hat)
  return(out)
}
bpriorMat <- function(delta.hat) {
  m <- rowMeans2(delta.hat)
  s2 <- rowVars(delta.hat)
  out <- (m*s2+m^3)/s2
  names(out) <- rownames(delta.hat)
  return(out)
}
postmean <- function(g.hat, g.bar, n, d.star, t2){
  (t2*n*g.hat+d.star*g.bar)/(t2*n+d.star)
}
postvar <- function(sum2, n, a, b){
  (.5*sum2+b)/(n/2+a-1)
}

# Helper function for parametric adjustements:
it.sol  <- function(sdat, g.hat, d.hat, g.bar, t2, a, b, conv=.0001){
	#n <- apply(!is.na(sdat),1,sum)
	n <- rowSums(!is.na(sdat))
	g.old  <- g.hat
	d.old  <- d.hat
	change <- 1
	count  <- 0
	ones <- rep(1,ncol(sdat))

	while(change>conv){
		g.new  <- postmean(g.hat,g.bar,n,d.old,t2)
		sum2   <- rowSums2((sdat-tcrossprod(g.new, ones))^2, na.rm=TRUE)
		d.new  <- postvar(sum2,n,a,b)
		change <- max(abs(g.new-g.old)/g.old,abs(d.new-d.old)/d.old)
		g.old <- g.new
		d.old <- d.new
		count <- count+1
		}
	adjust <- rbind(g.new, d.new)
	rownames(adjust) <- c("g.star","d.star")
	adjust
}



# Helper function for non-parametric adjustements:
int.eprior <- function(sdat, g.hat, d.hat){
    g.star <- d.star <- NULL
    r <- nrow(sdat)
    for(i in 1:r){
        g <- g.hat[-i]
        d <- d.hat[-i]		
        x <- sdat[i,!is.na(sdat[i,])]
        n <- length(x)
        j <- numeric(n)+1
        dat <- matrix(as.numeric(x), length(g), n, byrow=TRUE)
        resid2 <- (dat-g)^2
        sum2 <- resid2 %*% j
        LH <- 1/(2*pi*d)^(n/2)*exp(-sum2/(2*d))
        LH[LH=="NaN"]=0
        g.star <- c(g.star, sum(g*LH)/sum(LH))
        d.star <- c(d.star, sum(d*LH)/sum(LH))
        ## if(i%%1000==0){cat(i,'\n')}
    }
    adjust <- rbind(g.star,d.star)
    rownames(adjust) <- c("g.star","d.star")
    adjust	
} 


getNaiveEstimators <- function(s.data, dataDict, hasNAs, mean.only){
    batch.design <- dataDict$batch.design
    batches <- dataDict$batches
    if (!hasNAs){
        gamma.hat <- tcrossprod(solve(crossprod(batch.design, batch.design)), batch.design)
        gamma.hat <- tcrossprod(gamma.hat, s.data)
    } else{
        gamma.hat <- apply(s.data, 1, .betaNA, batch.design) 
    }
    delta.hat <- NULL
    for (i in dataDict$batches){
      if (mean.only){
        delta.hat <- rbind(delta.hat,rep(1,nrow(s.data))) 
      } else {
        delta.hat <- rbind(delta.hat,rowVars(s.data, cols=i, na.rm=TRUE))
      }    
    }
    colnames(gamma.hat)  <- colnames(delta.hat) <- rownames(s.data)
    rownames(gamma.hat)  <- rownames(delta.hat) <- names(batches)
    return(list(gamma.hat=gamma.hat, delta.hat=delta.hat))
}


getEbEstimators <- function(naiveEstimators,
      s.data, 
      dataDict,
      parametric=TRUE, 
      mean.only=FALSE
){
      gamma.hat=naiveEstimators[["gamma.hat"]]
      delta.hat=naiveEstimators[["delta.hat"]]
      batches=dataDict$batches
      n.batch=dataDict$n.batch
      ref.batch=dataDict$ref.batch
      ref=dataDict$ref
      .getParametricEstimators <- function(){
            gamma.star <- delta.star <- NULL
            for (i in 1:n.batch){
                if (mean.only){
                  gamma.star <- rbind(gamma.star,postmean(gamma.hat[i,], gamma.bar[i], 1, 1, t2[i]))
                  delta.star <- rbind(delta.star,rep(1, nrow(s.data)))
                } else {
                  temp <- it.sol(s.data[,batches[[i]]],gamma.hat[i,],delta.hat[i,],gamma.bar[i],t2[i],a.prior[i],b.prior[i])
                  gamma.star <- rbind(gamma.star,temp[1,])
                  delta.star <- rbind(delta.star,temp[2,])
                }
            }
            rownames(gamma.star) <- rownames(delta.star) <- names(batches)
            return(list(gamma.star=gamma.star, delta.star=delta.star))
      }
      .getNonParametricEstimators <- function(){
          gamma.star <- delta.star <- NULL
          for (i in 1:n.batch){
              if (mean.only){
                  delta.hat[i, ] = 1
              }
              temp <- int.eprior(as.matrix(s.data[, batches[[i]]]),gamma.hat[i,], delta.hat[i,])
              gamma.star <- rbind(gamma.star,temp[1,])
              delta.star <- rbind(delta.star,temp[2,])
          }
          rownames(gamma.star) <- rownames(delta.star) <- names(batches)
          return(list(gamma.star=gamma.star, delta.star=delta.star))
      }
      gamma.bar <- rowMeans(gamma.hat, na.rm=TRUE)
      t2 <- rowVars(gamma.hat, na.rm=TRUE)
      names(t2) <- rownames(gamma.hat)
      a.prior <- apriorMat(delta.hat)
      b.prior <- bpriorMat(delta.hat)
      if (parametric){
        temp <- .getParametricEstimators()
      } else {
        temp <- .getNonParametricEstimators()
      }
      if(!is.null(ref.batch)){
        temp[["gamma.star"]][ref,] <- 0  ## set reference batch mean equal to 0
        temp[["delta.star"]][ref,] <- 1  ## set reference batch variance equal to 1
      }
      out <- list()
      out[["gamma.star"]] <- temp[["gamma.star"]]
      out[["delta.star"]] <- temp[["delta.star"]]
      out[["gamma.bar"]] <- gamma.bar
      out[["t2"]] <- t2
      out[["a.prior"]] <- a.prior
      out[["b.prior"]] <- b.prior
      return(out)
}


getNonEbEstimators <- function(naiveEstimators, dataDict){
  out <- list()
  out[["gamma.star"]] <- naiveEstimators[["gamma.hat"]]
  out[["delta.star"]] <- naiveEstimators[["delta.hat"]]
  out[["gamma.bar"]]  <- NULL
  out[["t2"]] <- NULL
  out[["a.prior"]] <- NULL
  out[["b.prior"]] <- NULL
  ref.batch=dataDict$ref.batch
  ref=dataDict$ref
  if(!is.null(ref.batch)){
    out[["gamma.star"]][ref,] <- 0  ## set reference batch mean equal to 0
    out[["delta.star"]][ref,] <- 1  ## set reference batch variance equal to 1
  }
  return(out)
}


getCorrectedData <- function(dat, 
  s.data, 
  dataDict, 
  estimators, 
  naiveEstimators,
  stdObjects,
  eb=TRUE
){
  var.pooled=stdObjects$var.pooled
  stand.mean=stdObjects$stand.mean
  mod.mean=stdObjects$mod.mean
  batches <- dataDict$batches
  batch.design <- dataDict$batch.design
  n.batches <- dataDict$n.batches
  n.array <- dataDict$n.array
  ref.batch <- dataDict$ref.batch
  ref <- dataDict$ref
  if (eb){
    gamma.star <- estimators[["gamma.star"]]
    delta.star <- estimators[["delta.star"]]
  } else {
    gamma.star <- naiveEstimators[["gamma.hat"]]
    delta.star <- naiveEstimators[["delta.hat"]]
  }
  bayesdata <- s.data
  j <- 1
  for (i in batches){
      top <- bayesdata[,i]-t(batch.design[i,]%*%gamma.star)
      bottom <- tcrossprod(sqrt(delta.star[j,]), rep(1,n.batches[j]))
      bayesdata[,i] <- top/bottom
      j <- j+1
  }
  bayesdata <- (bayesdata*(tcrossprod(sqrt(var.pooled), rep(1,n.array))))+stand.mean+mod.mean
  if(!is.null(ref.batch)){
        bayesdata[, batches[[ref]]] <- dat[, batches[[ref]]]
  }
  return(bayesdata)
}


########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################

library(matrixStats)

data_path <- "datasets/small_test/"
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


# 
# 
# step1_data <- load("evaluation/d_combat/intermediate/GSE58135_step1.Rdata")
# 
# 
# step1_central_data <- load("evaluation/d_combat/intermediate/central_step1.Rdata")
# 
# step2_data <- load("evaluation/d_combat/intermediate/GSE58135_step2.Rdata")
# 
# 
step2_central_data <- load("evaluation/d_combat/intermediate/central_step2.Rdata")
# 

data <- list()
design <- list()
batches <- list()

data_path <- "datasets/small_test/"


distributedCombat_central(c(
  "evaluation/d_combat/intermediate/GSE58135_step2.Rdata",
  "evaluation/d_combat/intermediate/GSE129508_step2.Rdata",
  "evaluation/d_combat/intermediate/GSE149276_step2.Rdata"
),
file = paste0("intermediate/central_step2.Rdata")
)


for(lab in c("GSE129508" )){#, "GSE149276", "GSE58135")){
  
  dat <- read.table(paste0(data_path, "before/", lab, "/expr_for_correction.tsv"), 
                    header = TRUE, sep = "\t", row.names = 1)
  metadata <- read.table(paste0(data_path, "before/", lab, "/design.tsv"), 
                         header = TRUE, sep = "\t")
  
  # if rows in data contains only zeros, remove them
  # dat <- dat[rowSums(dat) != 0, ]
  dat <- dat[global_row_names, ]
  
  mod <- metadata$lum %>% as.matrix()
  rownames(mod) <- metadata$sample_id
  bat <- factor(metadata$batch, levels = c("0", "1", "2"))
  dat <- dat[, metadata$sample_id] %>% as.matrix()
  
  corrected_data_ind <- distributedCombat_site(dat, bat, mod,
                                               central.out = "evaluation/d_combat/intermediate/central_step2.Rdata"
  )$dat.combat %>% as.data.frame() %>% rownames_to_column("rowname")
  
}

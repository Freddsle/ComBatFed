
library(tidyverse)
library(gridExtra)
library(patchwork)
library(grid)
library(umap)
library(ggsci)

library(viridis)


pca_plot <- function(
    df, 
    batch_info, 
    title, 
    path = "", 
    quantitative_col_name = "Quantitative.column.name", 
    col_col = "Group", 
    shape_col = "",
    pc_x = "PC1",  # Default principal component for the x-axis
    pc_y = "PC2",  # Default principal component for the y-axis
    show_legend = TRUE,
    cbPalette = NULL
    ){
  pca <- prcomp(t(na.omit(df)))
  pca_df <- pca$x %>%
    as.data.frame() %>%
    rownames_to_column(quantitative_col_name) %>% 
    left_join(batch_info, by = quantitative_col_name)
  var_expl <- pca$sdev^2 / sum(pca$sdev^2)
  names(var_expl) <- paste0("PC", 1:length(var_expl))

  # Update the ggplot function call to use dynamic PC columns
  pca_plot <- pca_df %>%
      ggplot(aes_string(x = pc_x, y = pc_y, color = col_col, shape = shape_col)) +

  if(shape_col != ""){
    if(length(unique(batch_info[[shape_col]])) > 6){
      shapes_codes <- c(0, 1, 3, 8, 7, 15, 19)
      pca_plot <- pca_plot + 
        scale_shape_manual(values = shapes_codes)
    }    
  }

  pca_plot <- pca_plot + 
    geom_point(size=2) +
    theme_classic() +
    labs(title = title,
         x = glue::glue("{pc_x} [{round(var_expl[pc_x]*100, 2)}%]"),
         y = glue::glue("{pc_y} [{round(var_expl[pc_y]*100, 2)}%]"))

  if(!show_legend){
    pca_plot <- pca_plot + 
      theme(legend.position = "none")
  }

  if (!is.null(cbPalette)) {
    pca_plot <- pca_plot + scale_color_manual(values = cbPalette)
  }

  if (path == "") {
    return(pca_plot)
  } else {
    ggsave(path, pca_plot, width = 5, height = 5)
    return(pca_plot)
  }
}



umap_plot <- function(
  df, metadata, 
  title = "UMAP Projection", 
  color_column = "study_accession", 
  quantitative_col_name = 'sample',
  path = "") {
  # Perform UMAP on the transposed data
  umap_result <- umap(t(na.omit(df)))
  
  # Convert the UMAP result into a data frame and merge with metadata
  umap_data <- umap_result$layout %>%
    as.data.frame() %>%
    setNames(c("X1", "X2")) %>% 
    rownames_to_column(quantitative_col_name) %>% 
    left_join(metadata, by = quantitative_col_name) %>%
    column_to_rownames(quantitative_col_name)

  plot_result <- ggplot(umap_data, aes_string(x = "X1", y = "X2", color = color_column)) +
    geom_point(aes_string(col = color_column), size = 0.7) +
    # stat_ellipse(type = "t", level = 0.95) + # Add ellipses for each condition
    theme_minimal() +
    scale_color_lancet() + 
    labs(title = title, x = "UMAP 1", y = "UMAP 2") +
    guides(color = guide_legend(override.aes = list(size = 3))) # Ensure legend accurately represents centroids
  
    
    if (path == "") {
        return(plot_result)
  } else {
        ggsave(path, plot_result)
  }
}

# boxplot
boxplot_plot <- function(matrix, metadata_df, quantitativeColumnName, color_col, title, path="",
                         remove_xnames=FALSE) {
  # Reshape data into long format
  long_data <- tidyr::gather(matrix, 
                             key = "file", value = "Intensity")
  merged_data <- merge(long_data, metadata_df, by.x = "file", by.y = quantitativeColumnName)
  
  # Log tranformed scale
  boxplot <- ggplot(merged_data, aes(x = file, y = Intensity, fill = .data[[color_col]])) + 
    geom_boxplot() +
    stat_summary(fun = mean, geom = "point", shape = 4, size = 3, color = "red") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
    # adjust fonsize for the x-axis
    theme(axis.text.x = element_text(size = 8)) +
    labs(title = title) 

  if(remove_xnames){
    boxplot <- boxplot + theme(axis.title.x=element_blank(), axis.text.x=element_blank(), axis.ticks.x=element_blank())
  }

  if(path == "") {
    return(boxplot)
  } else {
      ggsave(path, boxplot)
      return(boxplot)
  }
}

boxplot_plot_groupped <- function(matrix, metadata_df, quantitativeColumnName, color_col, title, path="",
                         remove_xnames=FALSE, y_limits=NULL,
                         show_legend=TRUE, cbPalette=NULL) {
                          
  
  # Reshape data into long format and group by color_col
  long_data <- tidyr::gather(matrix, key = "file", value = "Intensity")
  merged_data <- merge(long_data, metadata_df, by.x = "file", by.y = quantitativeColumnName)
  
  # Group by color_col
  merged_data_grouped <- merged_data %>%
    group_by(.data[[color_col]])
  
  # Log transformed scale
  boxplot <- ggplot(merged_data_grouped, aes(x = .data[[color_col]], y = Intensity, fill = .data[[color_col]])) + 
    geom_violin(trim = F) +
    stat_summary(fun = median, geom = "crossbar", width = 0.35, color = "black", position = position_dodge(width = 0.2)) +
    stat_summary(fun = mean, geom = "point", shape = 4, size = 3, color = "darkred") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
    # adjust font size for the x-axis
    theme(axis.text.x = element_text(size = 8)) +
    labs(title = title) +
    guides(fill = guide_legend(override.aes = list(shape = NA, linetype = 0)))

  if(remove_xnames){
    boxplot <- boxplot + theme(axis.title.x=element_blank(), axis.text.x=element_blank(), axis.ticks.x=element_blank())
  }

  if (!is.null(y_limits)) {
    boxplot <- boxplot + ylim(y_limits)
  }

  if (!is.null(cbPalette)) {
    boxplot <- boxplot + scale_fill_manual(values = cbPalette)
  }


  if(!show_legend){
    boxplot <- boxplot + theme(legend.position = "none")
  }

  if(path == "") {
    return(boxplot)
  } else {
      ggsave(path, boxplot)
      return(boxplot)
  }
}



plotIntensityDensity <- function(
    intensities_df, metadata_df, quantitativeColumnName, colorColumnName, title, path=""
) {
  # Reshape the intensities_df from wide to long format
  long_intensities <- reshape2::melt(intensities_df, 
    variable.name = "Sample", value.name = "Intensity")
  
  # Adjust the merge function based on your metadata column names
  merged_data <- merge(long_intensities, metadata_df, by.x = "Sample", by.y = quantitativeColumnName)
  
  # Plot the data
  results <- ggplot(merged_data, aes(x = Intensity, color = .data[[colorColumnName]])) +  
    geom_density() +
    theme_minimal() +
    labs(title = paste(title, " by", colorColumnName),
         x = "Intensity",
         y = "Density")

  if(path == "") {
    return(results)
  } else {
    ggsave(path, results)
    return(results)
  }
}


heatmap_plot <- function(pg_matrix, batch_info, name, condition="condition", lab="lab"){
    cor_matrix <- cor(na.omit(pg_matrix), use = "pairwise.complete.obs")
    resulting_plot <- ggpubr::as_ggplot(grid::grid.grabExpr(
        pheatmap::pheatmap(cor_matrix, 
                        annotation_col = select(batch_info, c(condition, lab)),
                        treeheight_row = 0, treeheight_col = 0, 
                        main = paste0(name, ' heatmap')
        )
      )
    )
    return(resulting_plot)
}

plots_three <- function(expr, metadata, dataset, color_col='Status') {
    pca_plot <- pca_plot(
        expr, metadata, 
        dataset,
        col_col = color_col, shape_col = "Dataset", quantitative_col_name = 'sample_id')

    boxplot <- boxplot_plot(
        expr, metadata, 
        title = dataset,
        color_col = color_col, quantitativeColumnName = 'sample_id', 
        path = '')

    density_plot <- plotIntensityDensity(
        expr, metadata, 
        quantitativeColumnName = 'sample_id', 
        colorColumnName = color_col,
        title = dataset)

    return(list(pca_plot, density_plot, boxplot))
}


plot_diagnostic <- function(expression, metadata, dataset,
                            log_transform = FALSE, with_rowname = FALSE,
                            rownames_name = "gene_id") {
    
    plt_expression <- expression

    # if without rowname, then use the first column as rowname
    if (!with_rowname) {
        plt_expression <- plt_expression %>% column_to_rownames(rownames_name)
    } 
    if (!log_transform) {
        plt_expression <- log2(plt_expression + 1)
    }

    plt_meta <- metadata[metadata$sample_id %in% colnames(plt_expression),] %>%
        mutate(Status = as.factor(Status))
    plt_expression <- plt_expression[, plt_meta$sample_id]

    print("..plotting..")
    if (nrow(plt_expression) >10000) {
            #  plot
      plot_res <- plots_three(
          plt_expression[sample(1:nrow(plt_expression), 10000),], 
          plt_meta, 
          paste0(dataset, " log2(data + 1)")
          )
    } else {
      plot_res <- plots_three(
          plt_expression, 
          plt_meta, 
          paste0(dataset, " log2(data + 1)")
          )
    }


    return(plot_res)
}
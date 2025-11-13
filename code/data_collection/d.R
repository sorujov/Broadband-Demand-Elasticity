# ============================================================================
# ITU Data Download Script
# ============================================================================
# Purpose: Download telecommunications indicators from ITU DataHub API
# Author: Samir Orujov
# Date: November 13, 2025
# 
# Data Sources:
# - ITU DataHub API: https://datahub.itu.int/
# - Indicators: Fixed broadband, mobile subscriptions, internet users, 
#               bandwidth, and price baskets
# ============================================================================

# Load required libraries
library(httr)
library(readr)
library(dplyr)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Create data/raw directory if it doesn't exist
data_raw_dir <- "data/raw"
if (!dir.exists(data_raw_dir)) {
  dir.create(data_raw_dir, recursive = TRUE)
  cat("✓ Created directory:", data_raw_dir, "\n")
}

# Define target indicators with their ITU code IDs
indicators <- list(
  list(
    name = "fixed_broadband_subs",
    code_id = 19303,
    is_collection = FALSE,
    description = "Fixed-broadband subscriptions"
  ),
  list(
    name = "mobile_subs",
    code_id = 178,
    is_collection = FALSE,
    description = "Mobile-cellular subscriptions"
  ),
  list(
    name = "internet_users_pct",
    code_id = 11624,
    is_collection = FALSE,
    description = "Individuals using the Internet"
  ),
  list(
    name = "int_bandwidth",
    code_id = 242,
    is_collection = FALSE,
    description = "International bandwidth usage"
  ),
  list(
    name = "fixed_broad_price",
    code_id = 34616,
    is_collection = FALSE,
    description = "Fixed-broadband Internet basket"
  ),
  list(
    name = "mobile_broad_price",
    code_id = 34617,
    is_collection = FALSE,
    description = "Data-only mobile broadband basket"
  )
)

# ============================================================================
# FUNCTION: Download ITU Indicator Data
# ============================================================================

download_itu_indicator <- function(code_id, is_collection = FALSE, filename = NULL) {
  
  url <- paste0(
    "https://api.datahub.itu.int/v2/data/download/byid/",
    code_id,
    "/iscollection/",
    tolower(as.character(is_collection))
  )
  
  cat("  → Downloading from API...\n")
  
  # Download as ZIP file
  temp_zip <- tempfile(fileext = ".zip")
  
  response <- GET(
    url, 
    timeout(60),
    write_disk(temp_zip, overwrite = TRUE)
  )
  
  if (status_code(response) == 200) {
    
    # Check if it's actually a ZIP file or plain CSV
    file_type <- http_type(response)
    
    if (grepl("zip", file_type, ignore.case = TRUE) || 
        tools::file_ext(temp_zip) == "zip") {
      
      # It's a ZIP file - extract it
      temp_dir <- tempdir()
      
      tryCatch({
        unzip(temp_zip, exdir = temp_dir)
        
        # Find CSV files
        csv_files <- list.files(temp_dir, pattern = "\\.csv$", full.names = TRUE)
        
        if (length(csv_files) > 0) {
          data <- read_csv(csv_files[1], show_col_types = FALSE)
          cat("  ✓ Downloaded", nrow(data), "rows from ZIP\n")
          
          # Clean up temp files
          unlink(temp_zip)
          unlink(csv_files)
        } else {
          cat("  ✗ No CSV found in ZIP\n")
          unlink(temp_zip)
          return(NULL)
        }
      }, error = function(e) {
        cat("  ✗ Error extracting ZIP:", e$message, "\n")
        unlink(temp_zip)
        return(NULL)
      })
      
    } else {
      # It's a plain CSV
      data <- read_csv(temp_zip, show_col_types = FALSE)
      cat("  ✓ Downloaded", nrow(data), "rows (plain CSV)\n")
      unlink(temp_zip)
    }
    
    # Save to file if requested
    if (!is.null(filename)) {
      write_csv(data, filename)
      cat("  ✓ Saved to:", filename, "\n")
    }
    
    return(data)
    
  } else {
    cat("  ✗ HTTP status:", status_code(response), "\n")
    unlink(temp_zip)
    return(NULL)
  }
}

# ============================================================================
# MAIN EXECUTION: Download All Indicators
# ============================================================================

cat("================================================================================\n")
cat("ITU DATA DOWNLOAD\n")
cat("================================================================================\n")
cat("Date:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("Output directory:", data_raw_dir, "\n")
cat("Total indicators:", length(indicators), "\n\n")

# Store results
results <- list()
successful_downloads <- 0
failed_downloads <- 0

# Download each indicator
for (i in seq_along(indicators)) {
  indicator <- indicators[[i]]
  
  cat("--------------------------------------------------------------------------------\n")
  cat(sprintf("[%d/%d] %s\n", i, length(indicators), indicator$description))
  cat("--------------------------------------------------------------------------------\n")
  cat("  Code ID:", indicator$code_id, "\n")
  cat("  Variable name:", indicator$name, "\n")
  
  # Construct filename
  filename <- file.path(data_raw_dir, paste0("itu_", indicator$name, ".csv"))
  
  # Download data
  data <- download_itu_indicator(
    code_id = indicator$code_id,
    is_collection = indicator$is_collection,
    filename = filename
  )
  
  # Store result
  if (!is.null(data)) {
    results[[indicator$name]] <- data
    successful_downloads <- successful_downloads + 1
    
    # Display sample of data
    cat("\n  Preview (first 3 rows):\n")
    print(head(data, 3), n = 3)
  } else {
    results[[indicator$name]] <- NULL
    failed_downloads <- failed_downloads + 1
  }
  
  cat("\n")
  
  # Rate limiting - wait 2 seconds between requests
  if (i < length(indicators)) {
    Sys.sleep(2)
  }
}

# ============================================================================
# SUMMARY
# ============================================================================

cat("================================================================================\n")
cat("DOWNLOAD COMPLETE\n")
cat("================================================================================\n")
cat("✓ Successfully downloaded:", successful_downloads, "/", length(indicators), "\n")

if (successful_downloads > 0) {
  cat("\nDownloaded files:\n")
  for (indicator in indicators) {
    if (!is.null(results[[indicator$name]])) {
      filename <- paste0("itu_", indicator$name, ".csv")
      cat("  ✓", filename, "-", nrow(results[[indicator$name]]), "rows\n")
    }
  }
}

if (failed_downloads > 0) {
  cat("\n✗ Failed to download:", failed_downloads, "indicators\n")
  cat("\nFailed indicators:\n")
  for (indicator in indicators) {
    if (is.null(results[[indicator$name]])) {
      cat("  •", indicator$description, "(Code:", indicator$code_id, ")\n")
    }
  }
  cat("\nYou can try manual download from: https://datahub.itu.int/\n")
}

cat("\n================================================================================\n")
cat("All files saved to:", data_raw_dir, "\n")
cat("================================================================================\n")

# ============================================================================
# OPTIONAL: Create metadata file
# ============================================================================

metadata <- data.frame(
  variable_name = sapply(indicators, function(x) x$name),
  code_id = sapply(indicators, function(x) x$code_id),
  description = sapply(indicators, function(x) x$description),
  download_date = format(Sys.Date(), "%Y-%m-%d"),
  rows_downloaded = sapply(indicators, function(x) {
    if (!is.null(results[[x$name]])) nrow(results[[x$name]]) else NA
  }),
  status = sapply(indicators, function(x) {
    if (!is.null(results[[x$name]])) "Success" else "Failed"
  }),
  stringsAsFactors = FALSE
)

metadata_file <- file.path(data_raw_dir, "itu_download_metadata.csv")
write_csv(metadata, metadata_file)
cat("✓ Metadata saved to:", metadata_file, "\n")

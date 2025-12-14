#GWCSM Galaxy Analysis Tool

A GUI application for analyzing gravitational wave patterns in galaxy distributions, designed to support Gravitational Wave Cosmic Structure Mapping (GWCSM) research.

[GWCSM Analysis Tool]
*3D visualization of 500,000 galaxies showing cosmic structure*

## Overview

This tool enables researchers to:
- Load and analyze large galaxy datasets from SDSS
- Visualize 3D galaxy distributions in comoving coordinates
- Generate redshift histograms showing wave patterns
- Automatically detect peaks, troughs, and wavelengths in galaxy clustering
- Calculate statistical significance of observed structures

Built to facilitate research into gravitational wave interference as a mechanism for cosmic structure formation, as proposed in the GWCSM framework.

## Features

- **Interactive 3D Scatter Plots**: Rotate and explore galaxy distributions in 3D space
- **Redshift Analysis**: Automatically bin galaxies and identify wave patterns
- **Wave Pattern Detection**: Find peaks, troughs, and measure wavelengths between structures
- **Statistical Significance Testing**: Calculate sigma values for observed structures
- **Flexible Filtering**: Analyze specific sky regions or the complete dataset
- **Data Export**: Save filtered datasets for external analysis

## Installation

### Requirements
- Python 3.7 or higher
- Required packages:
pip install pandas numpy matplotlib

Setup

	1. Clone or download this repository
  
	2. Install dependencies (see above)
  
	3. Place galaxy data CSV in the galaxy_data/ folder
  
	4. Run the application: python gwcsm_galaxy_gui.py

Usage

	Loading Data
		1. Launch the application
		2. Go to Load Existing CSV tab
		3. Browse to your galaxy data CSV file
		4. Select filtering mode:
			• Whole Sky: Use all data in the file
			• Specific Slice: Filter by RA, Dec, and redshift ranges
				1. Click Apply Filter & Load Data
	Analysis
		Once data is loaded, switch to the Analysis & Visualization tab:
			• Redshift Histogram: Shows galaxy count distribution across redshift bins
			• 3D Scatter Plot: Interactive visualization of galaxy positions (drag to rotate, scroll to zoom)
			• Wave Analysis: Automatically detects peaks, troughs, wavelengths, and statistical significance (gives an error, disregard warning)
	Interpreting Results
		The Wave Analysis output includes:
			• Peaks: Redshift values where galaxy concentration exceeds background (constructive interference)
			• Troughs: Redshift values where galaxy density is below background (destructive interference)
			• Wavelength: Distance between consecutive peaks (in Mpc)
			• Statistical Significance: Sigma values indicating how significant the patterns are
	Included Dataset
		The repository includes The_One_To_View_Them_All.csv containing:
			• 500,000 galaxies from SDSS DR18
			• Redshift range: z = 0.15 to 0.65
			• Sky coverage: Complete SDSS spectroscopic survey footprint
			• Columns: objID, ra, dec, z, type
		This dataset represents all SDSS spectroscopic galaxies in the specified redshift range and provides a comprehensive view of cosmic structure for GWCSM analysis.

Getting Additional Data

	To download more galaxy data:
		1. Visit SDSS SkyServer SQL Search
		2. Use this query template:
    Select
        p.objid, p.ra, p.dec, s.z, s.zerr, p.petroMag_r
    FROM 
        PhotoObj AS p
    JOIN
        SpecObj AS s ON s.bestobjid = p.objid
    WHERE
        p.ra BETWEEN # AND #
        AND p.dec BETWEEN # AND #
        AND s.z BETWEEN 0.15 AND 0.65
        AND s.zwarning = 0
        AND s.class = 'GALAXY'

		3. Save as CSV and load in the application 
    
***Be sure to edit the CSV to get rid of the first line that reads "#table"***

	Example of how the CSV should start: 

    objid,ra,dec,z,zerr,petroMag_r
    1237645942905897152,58.0630138368383,0.0881682357798876,0.1654301,4.361457E-05,17.71811
    1237645942905897517,58.0580008820462,0.176158361845024,0.240065,3.420843E-05,19.08798

“One CSV to rule them all, One tool to find them, One GUI to bring them all, and in the analysis bind them.”

#!/bin/bash

# ---------------------------------------------------------------
# Script Name: SVG to ICO Converter
# Description: This bash script automates the process of converting 
#              an SVG (Scalable Vector Graphics) file to an ICO 
#              (Icon) file. It uses the tools Inkscape and ImageMagick. 
#              The script takes an SVG file 'logo.svg', scales it to a 
#              128x128 PNG image using Inkscape, then converts the PNG 
#              image to an ICO file using ImageMagick, and finally removes
#              the temporary PNG file.
# 
# Prerequisites: Please ensure Inkscape and ImageMagick are installed 
#                on your system. For Arch Linux and its derivatives, you 
#                can use the following command: 
#                sudo pacman -S inkscape imagemagick
#
# Usage: To run the script, navigate to the directory containing the script 
#        and the 'logo.svg' file, then execute the script. In a terminal, 
#        you can do this by typing:
#                sh ./generate-favicon.sh
# ---------------------------------------------------------------

# Export the SVG image to a temporary PNG image with Inkscape
inkscape -w 128 -h 128 -o temp.png logo.svg

# Convert the PNG image to ICO with ImageMagick
convert temp.png favicon.ico

# Remove the temporary PNG image
rm temp.png

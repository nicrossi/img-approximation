# Genetic Image Compressor
An image “compressor” that approximates any input picture by layering semi-transparent triangles on a white canvas. 
A custom Genetic Algorithms engine evolves candidate artworks—each genome encodes triangle count, positions, colors, and alpha—optimizing a fitness function to minimize the visual error vs. the target image. 

The system supports pluggable selection/crossover/mutation strategies and YAML/JSON configs for reproducible runs. Outputs include the best-rendered image, evolution metrics (fitness over generations), 
and an image representation of the final triangle set—offering a compact, human-legible “compressed” form.

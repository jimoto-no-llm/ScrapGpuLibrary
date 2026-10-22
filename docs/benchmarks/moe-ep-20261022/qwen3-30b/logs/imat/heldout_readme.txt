coverage of corpus B traffic by the top-K set chosen on corpus A (mean over 48 layers; uniform = K/128)
A \ B       readme_h1  readme_h2   K
readme_h1   0.582      0.566       16 (uniform 0.125)
readme_h2   0.563      0.584       16 (uniform 0.125)
readme_h1   0.780      0.760       32 (uniform 0.250)
readme_h2   0.758      0.783       32 (uniform 0.250)
readme_h1   0.893      0.875       48 (uniform 0.375)
readme_h2   0.871      0.894       48 (uniform 0.375)
readme_h1   0.953      0.942       64 (uniform 0.500)
readme_h2   0.939      0.954       64 (uniform 0.500)

readme_h1 experts needed per layer (of 128): 50% of traffic mean 12.9 (min 7 max 28), 80% mean 34.7 (min 21 max 65)
readme_h2 experts needed per layer (of 128): 50% of traffic mean 12.8 (min 7 max 28), 80% mean 34.5 (min 22 max 64)

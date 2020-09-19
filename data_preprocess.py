import datasets.cvtransforms as transforms_dct 


input_size1 = 512
input_size2 = 448 
transform = transforms_dct.Compose([
            transforms_dct.Resize(input_size1),
            transforms_dct.CenterCrop(input_size2)),
            transforms_dct.Upscale(upscale_factor=2)
            ])



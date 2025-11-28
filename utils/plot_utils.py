def get_in_inch_dimension_given_target_h_in_pixel(img, target_height_px, dpi = 100):
    ## formula: inch = pixel / dpi
    h, w = img.shape
    aspect_ratio = w / h
    height_inch = target_height_px / dpi
    width_inch = height_inch * aspect_ratio
    return width_inch, height_inch
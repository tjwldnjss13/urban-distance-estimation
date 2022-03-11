function [ cm ] = read_colormap()

    % RGB values of labels IDs
    cm = [ 205, 133,  63;
             0, 150,  47;
           220,  20,  60;
           238, 232, 170;
            70, 130, 180;
           255, 255,   0;
           102, 102,   0;
           204,  51,  51;
             0,   0, 142;
            13, 217, 200;
            46,  35, 189 ];
        
     % fill up with zeros to obtain 255x3 matrix
     cm = [ cm; zeros(244, 3) ];
     
     % normalize to 0..1
     cm = cm ./ 255;

end
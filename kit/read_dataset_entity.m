function [I, D, L] = read_dataset_entity(seqDir, imgNr)

    % build filenames
    fname_image = strcat(seqDir,'/imgleft/imgleft',sprintf('%09d', imgNr),'.pgm');
    fname_disp  = strcat(seqDir,'/disp/disp',sprintf('%09d', imgNr),'.pgm');
    fname_label = strcat(seqDir,'/groundtruth/label',sprintf('%09d', imgNr),'.pgm');
    
    % read data
    I    = imread(fname_image);
    Draw = imread(fname_disp);
    L    = imread(fname_label);
    
    % convert raw disparity image to original float range
    D = single(Draw) ./ 512.0;

end
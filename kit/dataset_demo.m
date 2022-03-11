function dataset_demo()

    for seqDir = { 'train_1', 'train_2', 'train_3', 'test_1', 'test_2' }
        
        for imgNr = linspace(9, 999, 100)
            
            [image, disp, labels] = read_dataset_entity(seqDir{1}, imgNr);
            
            subplot(3,1,1);
            subimage(image);
            
            subplot(3,1,2);
            imshow(disp);
            colormap(jet);
            caxis([0, 50]);
            
            subplot(3,1,3);
            cm_labels = read_colormap();
            labelsRGB = label2rgb(labels+1, cm_labels);
            imshow(labelsRGB);
            
            pause(0.05);
            
        end
    end
    
end
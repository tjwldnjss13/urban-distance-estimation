function [ xImg, yImg, zImg ] = image_to_world_full(dispImage, seqDir)

    xImg = zeros(size(dispImage));
    yImg = zeros(size(dispImage));
    zImg = zeros(size(dispImage));

    for v=1:size(dispImage,1)
        for u=1:size(dispImage,2)

            [ x, y, z ] = image_to_world(u, v, dispImage(v,u), seqDir);

            % copy to result
            xImg(v,u) = x;
            yImg(v,u) = y;
            zImg(v,u) = z;

        end
    end

end
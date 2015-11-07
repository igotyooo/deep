function [ cls_error, clsloc_error ] = my_eval_ilsvrc_clsloc...
    ( idx2iid, idx2pred, gtruth_file, max_num_pred_per_image, blacklist_file, optional_cache_file )
    iid2gtcid = dlmread( gtruth_file );
    biids = dlmread( blacklist_file );
    iid2black = false( size( iid2gtcid, 1 ), 1 );
    iid2black( biids ) = true;
    iid2gtbox = load( optional_cache_file );
    iid2gtbox = iid2gtbox.rec;
    idx2gtcid = iid2gtcid( idx2iid );
    idx2black = iid2black( idx2iid );
    idx2gtbox = iid2gtbox( idx2iid );
    numIm = size( idx2pred, 1 );
    if size( idx2pred, 2 ) > max_num_pred_per_image * 5,
        idx2pred = idx2pred( :, 1 : max_num_pred_per_image * 5 ); end;
    num_guesses = size( idx2pred, 2 ) / 5;
    idx2predcid = idx2pred( :, 1 : 5 : end );
    % Compute classification error.
    x = idx2gtcid * ones( 1, size( idx2predcid, 2 ) );
    c = min( x ~= idx2predcid, [  ], 2 );
    cls_error = sum( c ) / numIm;
    % compute localization error
    numBlack = sum( idx2black );
    t = tic;
    idx2err = zeros( numIm, 1 );
    for idx = 1 : numIm
        if toc( t ) > 60,
            fprintf( 'EVALCLSLOC: on %i of %i\n', idx, numIm );
            t = tic;
        end;
        idx2err( idx ) = 0;
        if idx2black( idx ), continue; end;
        for j = 1 : num_guesses,
            d_jk = ( idx2gtcid( idx ) ~= idx2predcid( idx, j ) );
            if d_jk == 0,
                box = idx2pred( idx, ( j - 1 ) * 5 + 1 + ( 1 : 4 ) );
                ov_vector = compute_overlap( box, idx2gtbox( idx ), idx2gtcid( idx ) );
                f_j = ( ov_vector < 0.50 );
            else
                f_j = 1;
            end;
            d_jk = ones( 1, numel( f_j ) ) * d_jk;
            d( idx, j ) = min( max( [ f_j; d_jk ] ) );
            idx2err( idx ) = idx2err( idx ) + min( d( idx, : ) );
        end;
    end;
    clsloc_error = sum( idx2err ) / ( numIm - numBlack );
end
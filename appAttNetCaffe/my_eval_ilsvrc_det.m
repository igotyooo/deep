function [ ap, recall, precision ] = my_eval_ilsvrc_det...
    ( idx2iid, rid2pred, cid2numObj, meta_file, blacklist_file, optional_cache_file )
    data = load( meta_file );
    synsets = data.synsets;
    hash = make_hash( synsets );
    data = load( optional_cache_file );
    iid2iid = data.gt_img_ids;
    iid2cidsGt = data.gt_obj_labels;
    iid2bboxsGt = data.gt_obj_bboxes;
    iid2thrs = data.gt_obj_thr;
    
    idx2cidsGt = iid2cidsGt( idx2iid );
    idx2bboxsGt = iid2bboxsGt( idx2iid );
    idx2thrs = iid2thrs( idx2iid );
    
    iid2idx = zeros( size( iid2iid ) );
    iid2idx( idx2iid ) = 1 : numel( idx2iid );
    idx2idx = ( 1 : numel( idx2iid ) )';
    % Blacklist.
    [ bidx2iid, bidx2wnid ] = textread( blacklist_file, '%d %s' );
    bidx2ok = ismember( bidx2iid, idx2iid );
    bidx2iid = bidx2iid( bidx2ok );
    bidx2wnid = bidx2wnid( bidx2ok );
    bidx2cid = zeros( length( bidx2wnid ), 1 );
    for i = 1 : length( bidx2wnid ),
        bidx2cid( i ) = get_class2node( hash, bidx2wnid{ i } );
    end;
    bidx2idx = iid2idx( bidx2iid );
    % Reform prediction info.
    oid2iid = rid2pred( :, 1 );
    oid2cidPred = rid2pred( :, 2 );
    oid2scorePred = rid2pred( :, 3 );
    oid2bboxPred = rid2pred( :, 4 : 7 )';
    oid2idx = iid2idx( oid2iid );
    % Sort by image id.
    [ oid2idx, ind ] = sort( oid2idx );
    oid2scorePred = oid2scorePred( ind );
    oid2cidPred = oid2cidPred( ind );
    oid2bboxPred = oid2bboxPred( :, ind );
    % Compute per-image informations.
    endidx = max( max( idx2idx ), max( oid2idx ) );
    idx2cidxPred = cell( 1, endidx );
    idx2scoresPred = cell( 1, endidx );
    idx2bboxsPred = cell( 1, endidx );
    startoid = 1;
    idx = oid2idx( 1 );
    for oid = 1 : length( oid2idx ),
        if ( oid == length( oid2idx ) ) || ( oid2idx( oid + 1 ) ~= idx ),
            idx2cidxPred{ idx } = oid2cidPred( startoid : oid )';
            idx2scoresPred{ idx } = oid2scorePred( startoid : oid )';
            idx2bboxsPred{ idx } = oid2bboxPred( :, startoid : oid );
            if oid < length( oid2idx ), idx = oid2idx( oid + 1 ); startoid = oid + 1; end;
        end;
    end;
    for idx = 1 : endidx,
        [ idx2scoresPred{ idx }, ind ] = sort( idx2scoresPred{ idx }, 'descend' );
        idx2cidxPred{ idx } = idx2cidxPred{ idx }( ind );
        idx2bboxsPred{ idx } = idx2bboxsPred{ idx }( :, ind );
    end;
    % Compute TP/FP.
    idx2tps = cell( 1, endidx );
    idx2fps = cell( 1, endidx );
    numCls = length( cid2numObj );
    for idx = 1 : length( idx2idx ),
        cidsGt = idx2cidsGt{ idx };
        bboxsGt = idx2bboxsGt{ idx };
        thrs = idx2thrs{ idx };
        numObjGt = length( cidsGt );
        detected = zeros( 1, numObjGt );
        bcids = bidx2cid( bidx2idx == idx );
        cidsPred = idx2cidxPred{ idx };
        bboxsPred = idx2bboxsPred{ idx };
        numObjPred = length( cidsPred );
        tps = zeros( 1, numObjPred );
        fps = zeros( 1, numObjPred );
        for op = 1 : numObjPred
            if any( cidsPred( op ) == bcids ), continue; end;
            bb = bboxsPred( :, op );
            ovmax = -inf;
            kmax = -1;
            for og = 1 : numObjGt
                if cidsPred( op ) ~= cidsGt( og ), continue; end;
                if detected( og ) > 0, continue; end;
                bbgt = bboxsGt( :, og );
                bi = [ ...
                    max( bb( 1 ), bbgt( 1 ) ); ...
                    max( bb( 2 ), bbgt( 2 ) ); ...
                    min( bb( 3 ), bbgt( 3 ) ); ...
                    min( bb( 4 ), bbgt( 4 ) ) ];
                iw = bi( 3 ) - bi( 1 ) + 1;
                ih = bi( 4 ) - bi( 2 ) + 1;
                if iw > 0 && ih > 0,
                    ua = ( bb( 3 ) - bb( 1 ) + 1 ) * ( bb( 4 ) - bb( 2 ) + 1 ) +...
                        ( bbgt( 3 ) - bbgt( 1 ) + 1 ) * ( bbgt( 4 ) - bbgt( 2 ) + 1 ) - iw * ih;
                    ov = iw * ih / ua;
                    if ov >= thrs( og ) && ov > ovmax, ovmax = ov; kmax = og; end;
                end;
            end;
            if kmax > 0, tps( op ) = 1; detected( kmax ) = 1; else fps( op ) = 1; end;
        end;
        idx2tps{ idx } = tps;
        idx2fps{ idx } = fps;
        for og = 1 : numObjGt,
            cidGt = cidsGt( og );
            if any( cidGt == bcids ), cid2numObj( cidGt ) = cid2numObj( cidGt ) - 1; end;
        end;
    end;
    oid2tp = [ idx2tps{ : } ];
    oid2fp = [ idx2fps{ : } ];
    oid2cidPred = [ idx2cidxPred{ : } ];
    oid2scorePred = [ idx2scoresPred{ : } ];
    [ ~, rank2oid ] = sort( oid2scorePred, 'descend' );
    rank2tp = oid2tp( rank2oid );
    rank2fp = oid2fp( rank2oid );
    rank2cid = oid2cidPred( rank2oid );
    recall = cell( numCls, 1 );
    precision = cell( numCls, 1 );
    ap = zeros( numCls, 1 );
    for c = 1 : numCls,
        tp = cumsum( rank2tp( rank2cid == c ) );
        fp = cumsum( rank2fp( rank2cid == c ) );
        recall{ c } = ( tp / cid2numObj( c ) )';
        precision{ c } = ( tp ./ ( fp + tp ) )';
        ap( c ) = VOCap( recall{ c }, precision{ c } );
    end;
end
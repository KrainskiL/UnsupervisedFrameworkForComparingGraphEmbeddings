using Random
using DelimitedFiles
using Statistics

# Tested under Julia 1.6.2

distance(emb, i, j) =
    sqrt(sum((emb[i+1, 2:end] - emb[j+1, 2:end]).^ 2))

# the code assumes
# - nodes are labelled from 0
# - embedding is sorted on node id; node id can be ignored and is a first column
# - files are space delimited
function reshuffle_edges(edge_file::AbstractString,
                         community_file::AbstractString,
                         embedding_file::AbstractString,
                         out_edge_file::AbstractString,
                         shufflepct::Real;
                         seed::Int=1234,
                         check_correctness::Bool=true)
    edges_raw = readdlm(edge_file, Int)
    communities = readdlm(community_file, Int)
    embedding = readdlm(embedding_file, Float64)

    @assert size(edges_raw, 2) == 2
    vmin, vmax = extrema(edges_raw)
    @assert vmin >= 0
    @assert vmax < length(communities)
    @assert size(communities, 2) == 1
    @assert size(embedding,1) == length(communities)

    comm_edges = Dict{Int, Set{Tuple{Int,Int}}}()
    background = Set{Tuple{Int,Int}}()

    for (a, b) in eachrow(edges_raw)
        com = communities[a+1]
        if com == communities[b+1]
            comm_set = get!(comm_edges, com, Set{Tuple{Int, Int}}())
            @assert !((a, b) in comm_set)
            push!(comm_set, (a, b))
        else
            @assert !((a, b) in background)
            push!(background, (a, b))
        end
    end
    @assert sum(length.(values(comm_edges))) + length(background) == size(edges_raw, 1)

    Random.seed!(seed)
    for comm_set in values(comm_edges)
        if check_correctness
            avg_dist = mean(distance(embedding, a, b) for (a,b) in comm_set)
            @info "old avg distance: $avg_dist"
        end
        len = length(comm_set)
        must_rewire = len * shufflepct
        did_rewire = 0
        while did_rewire < must_rewire
            (a1, a2) = rand(comm_set)
            (b1, b2) = rand(comm_set)
            (a1, a2) == (b1, b2) && continue
            (a1, b2) == (b1, a2) && continue
            (a1, b2) in comm_set && continue
            (b1, a2) in comm_set && continue
            @assert a1 >= 0
            @assert a2 >= 0
            @assert b1 >= 0
            @assert b2 >= 0
            old_sum = distance(embedding, a1, a2) + distance(embedding, b1, b2)
            new_sum = distance(embedding, a1, b2) + distance(embedding, b1, a2)
            old_sum >= new_sum && continue
            pop!(comm_set, (a1, a2))
            pop!(comm_set, (b1, b2))
            push!(comm_set, (a1, b2))
            push!(comm_set, (b1, a2))
            did_rewire += 1
            @assert length(comm_set) == len
        end
        if check_correctness
            avg_dist = mean(distance(embedding, a, b) for (a,b) in comm_set)
            @info "new avg distance: $avg_dist"
            println()
        end
    end

    @assert sum(length.(values(comm_edges))) + length(background) == size(edges_raw, 1)
    open(out_edge_file, "w") do io
        for (a, b) in background
            println(io, a, " ", b)
        end
        for comm_set in values(comm_edges)
            for (a, b) in comm_set
                println(io, a, " ", b)
            end
        end
    end

    check_correctness || return

    edges_raw2 = readdlm(out_edge_file, Int)
    if iszero(shufflepct)
        @assert sort!(collect(eachrow(edges_raw))) ==
                sort!(collect(eachrow(edges_raw2)))
    else
        @assert sort!(edges_raw[:, 1]) == sort!(edges_raw2[:, 1])
        @assert sort!(edges_raw[:, 2]) == sort!(edges_raw2[:, 2])
    end
end

reshuffle_edges(ARGS[1],
                ARGS[2],
                ARGS[3],
                ARGS[4], parse(Float64,ARGS[5]))

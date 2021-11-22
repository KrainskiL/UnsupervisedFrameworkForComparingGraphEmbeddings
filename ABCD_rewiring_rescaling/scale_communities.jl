using Random
using DelimitedFiles
using Statistics

# Tested under Julia 1.6.2

function scale(center::Matrix{Float64}, point::Matrix{Float64}, scaling::Float64)
    @assert size(center) == size(point)
    @assert size(center, 1) == 1
    return scaling .* (point .- center) .+ center
end

# the code assumes
# - node id in embedding can be ignored and is a first column
# - communities are in the same order as nodes
# - files are space delimited
# - center of the community is computed using mean
function scale_communities(community_file::AbstractString,
                           embedding_file::AbstractString,
                           out_embedding_file::AbstractString,
                           scaling::Float64)
    communities = readdlm(community_file, Int)
    embedding = readdlm(embedding_file, Float64)[:, 2:end]

    @assert size(communities, 2) == 1
    @assert size(embedding, 1) == size(communities, 1)

    comm_ids = Dict{Int, Vector{Int}}()

    for (i, com) in enumerate(communities)
        push!(get!(comm_ids, com, Int[]), i)
    end

    @assert sum(length.(values(comm_ids))) == size(embedding, 1)
    @assert sort!(reduce(vcat, values(comm_ids))) == axes(embedding, 1)

    new_embedding = fill(NaN, size(embedding))
    for ids in values(comm_ids)
        center = mean(embedding[ids, :], dims=1)
        for i in ids
            @assert all(isnan, new_embedding[i, :])
            new_embedding[i:i, :] = scale(center, embedding[i:i, :], scaling)
            @assert !any(isnan, new_embedding[i, :])
        end
        @assert center ≈ mean(new_embedding[ids, :], dims=1)
        if scaling == 1.0
            @assert embedding[ids, :] ≈ new_embedding[ids, :]
        end
        if scaling == 0.0
            @assert repeat(center, length(ids)) ≈ new_embedding[ids, :]
        end
    end
    @assert !any(isnan, new_embedding)
    open(out_embedding_file, "w") do io
        for (i, row) in enumerate(eachrow(new_embedding))
            println(io, i-1, " ", join(row, " "))
        end
    end
    if scaling == 1
        @assert readdlm(embedding_file, Float64)[:, 2:end] ≈
                readdlm(out_embedding_file, Float64)[:, 2:end]
    end
    return nothing
end

scale_communities(ARGS[1],ARGS[2],ARGS[3], parse(Float64,ARGS[4]))

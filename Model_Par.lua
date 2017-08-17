require 'nn'
local net = nn.Sequential()

function net.CreateModel(ph,pw,kernels,strides,hiddens,h,w)
	local netProjection = nn.Sequential()
	kernels = torch.Tensor(kernels)
	hiddens = torch.Tensor(hiddens)
	strides = torch.Tensor(strides)
	local nLayers = kernels:size(1)-1 
	local drop = 0.2
	local d = kernels[1]
	local pad = 2
	local convStride = 1
	local new_h=h
	local new_w=w
	for i=1,nLayers do
		netProjection:add(nn.SpatialConvolution(kernels[i],kernels[i+1],ph,pw,convStride,convStride,pad,pad))
		netProjection:add(nn.ReLU())
		netProjection:add(nn.SpatialMaxPooling(strides[i],strides[i]))
		new_h = torch.floor((torch.floor((new_h-ph+2*pad)/convStride+1))/strides[i])
		new_w = torch.floor((torch.floor((new_w-pw+2*pad)/convStride+1))/strides[i])
	end
	local nFeatures = new_h*new_w*kernels[#kernels]*3
	
	local parNet = nn.Parallel(1,3)
	for p=1,3 do
		parNet:add(netProjection:clone())
	end
	
	net:add(parNet)
	net:add(nn.View(nFeatures))
	net:add(nn.Dropout(drop))
	net:add(nn.Linear(nFeatures,hiddens[1]))
	net:add(nn.ReLU())
	nHiddens = (torch.type(#hiddens)=='number') and #hiddens or (#hiddens)[1]
	for j=2,nHiddens do
		net:add(nn.Dropout(drop))
		net:add(nn.Linear(hiddens[j-1],hiddens[j]))
		net:add(nn.ReLU())
	end
	
	net:add(nn.Linear(hiddens[nHiddens],1))
	--net:add(nn.Sigmoid())
end

return net
	

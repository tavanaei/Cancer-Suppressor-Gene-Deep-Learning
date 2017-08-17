require 'nn';
require 'cunn';
require 'cutorch';
require 'nn';
--require 'cudnn';
require 'optim'
--require 'jxtools'
require 'paths'

torch.setnumthreads(8)
torch.setdefaulttensortype('torch.FloatTensor')

--local util = jxtools.util
local proc = dofile('Data.lua')
local Param = dofile('opts.lua')
local opt = Param.Parse(arg)
print(opt)
cutorch.setDevice(opt.GPU)

local dataNames,label = proc.DataCollect(opt.positive,opt.negative,opt.neutral)
local net = dofile('Model_Par.lua')
local h=200
local w=200

net.CreateModel(opt.p,opt.p,opt.kernel,opt.stride,opt.hidden,h,w)
local criterion = nn.MSECriterion() --nn.ClassNLLCriterion()
net = (opt.cuda) and net:cuda() or net
criterion = (opt.cuda) and criterion:cuda() or criterion
local optimState = {
    learningRate = opt.learningRate,
    learningRateDecay = opt.learningRateDecay,
    momentum = opt.momentum,
    dampening = 0.0,
    nesterov = true,
    alpha = opt.alpha or 0.95,
    weightDecay = opt.weightDecay
}
params,gradParams = net:getParameters()
local fetchSize = 600
local steps = torch.floor(fetchSize/opt.batchSize)
local mean = torch.zeros(opt.trainSize/fetchSize,16)
local stdv = torch.zeros(opt.trainSize/fetchSize,16)
local lossRec = torch.Tensor((fetchSize/opt.batchSize/100+1)*opt.trainSize/fetchSize*opt.iterations)
local validNames = proc.Slice(dataNames,opt.trainSize+1,opt.trainSize+opt.validSize)

local validProteins = proc.Fetch(validNames,h,w)
local validset,_,_ = proc.Dataset(validProteins,label[{{opt.trainSize+1,opt.trainSize+opt.validSize}}])
if (opt.cuda) then validset.data = validset.data:cuda() end
validNames = nil
validProteins = nil
collectgarbage()
trackIndex = 1

local FinalRes = torch.Tensor(opt.iterations)

local function Test(index)
	local testSize = opt.testSize
	local predictLab = torch.zeros(testSize)
	local target = torch.zeros(testSize)

	for r=1,testSize,10 do
		--print(r)
		local names = proc.Slice(dataNames,opt.trainSize+opt.validSize+r,opt.trainSize+opt.validSize+r+9)
		local pros = proc.Fetch(names,h,w)
		local testset = proc.Dataset(pros,label[{{opt.trainSize+opt.validSize+r,opt.trainSize+opt.validSize+r+9}}])
		if (opt.cuda) then testset.data = testset.data:cuda() end
		local predictProb = net:forward(testset.data):clone()
		for i=1,10 do
			predictLab[r+i-1]= (predictProb[i][1]>=0.5) and 1 or 0
			target[r+i-1] = testset.label[i]
			--print(predictProb)
		end
	end
	FinalRes[index] = torch.sum(target:eq(predictLab))/testSize
	print(torch.sum(target:eq(predictLab))/testSize)
end
print(" ********** TRAIN **********")
for iter=1,opt.iterations do
		if (iter%10==0 or iter==1) do
			print(iter)
			Test(iter)
		end
	for box=1,opt.trainSize/fetchSize do
		--print(" ............. OVERLAY ............")
		local st = (box-1)*fetchSize+1
		local ed = box*fetchSize
		local names = proc.Slice(dataNames,st,ed)
		local pros = proc.Fetch(names,h,w)
		local trainset={}
		local a={}
		local b={}
		trainset,a,b = proc.Dataset(pros,label[{{st,ed}}])
		mean[{ {box}, {} }] = a
		stdv[{ {box}, {} }] = b	
		pros = nil
		names = nil
		
		for t=1,fetchSize,opt.batchSize do
			local subData = (opt.cuda) and trainset.data[{{}, {t,t+opt.batchSize-1}, {}, {}, {} }]:cuda() or trainset.data[{{}, {t,t+opt.batchSize-1}, {}, {}, {} }]
			local subLabel= (opt.cuda) and trainset.label[{{t,t+opt.batchSize-1}}]:cuda() or trainset.label[{{t,t+opt.batchSize-1}}]

			function feval(params)
				gradParams:zero()
				local outputs = net:forward(subData):clone()
				local loss = criterion:forward(outputs,subLabel)
		
				local dloss_doutputs = criterion:backward(outputs,subLabel)
				net:backward(subData,dloss_doutputs)
				return loss, gradParams
			end
			optim.sgd(feval, params, optimState) 
		end
		subData = nil
		subLabel = nil
		collectgarbage()
	end
	
		
end

net:evaluate()

local testMean = mean:mean(1):reshape(16)
local testStdv = stdv:mean(1):reshape(16)
torch.save(opt.model,net)
torch.save('MeanStd.dat',{mean=testMean, stdv = testStdv})
mean=nil
stdv=nil
trainset=nil
collectgarbage()
print(" .......  Testing ....... ")
local testSize = opt.testSize
local ROC= 1
local predictLab = torch.zeros(testSize,ROC)
local target = torch.zeros(testSize)

for r=1,testSize,10 do
	--print(r)
	local names = proc.Slice(dataNames,opt.trainSize+opt.validSize+r,opt.trainSize+opt.validSize+r+9)
	local pros = proc.Fetch(names,h,w)
	local testset = proc.Dataset(pros,label[{{opt.trainSize+opt.validSize+r,opt.trainSize+opt.validSize+r+9}}],testMean,testStdv)
	if (opt.cuda) then testset.data = testset.data:cuda() end
	local predictProb = net:forward(testset.data):clone()
	for i=1,10 do
		target[r+i-1] = testset.label[i]
		for roc=1,ROC do
			predictLab[r+i-1][roc]= (predictProb[i][1]>=(roc*0.5)) and 1 or 0
		--print(predictProb)
		end
	end
end
--print(torch.sum(target:eq(predictLab))/opt.testSize)
torch.save(opt.result,torch.cat(target,predictLab,2))
torch.save(opt.model,net:float())
torch.save(opt.result .. "Final",FinalRes)
print("DONE!")


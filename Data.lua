local M={}

require 'paths';
torch.manualSeed(123)

local function LoadCSV(proteinName)
	local file = io.open(proteinName, 'r')
	local header = file:read()
	local dat = {}
	for l in file:lines() do
		local row = l:split(',')
		table.insert(dat,row)
	end
	return torch.Tensor(dat)
end

function M.Maper(cancerType)
	local pros = paths.dir('Data/' .. cancerType)
	os.execute("mkdir " .. "Data/" .. cancerType .. "_Map")
	table.remove(pros,1)
	table.remove(pros,1)
	table.sort(pros)
	local n = #pros
	
	for i=1,n,3 do
		local tens = {}
		print(pros[i])
		for j=1,3 do
			local map = torch.zeros(16,200,200)
			gene = LoadCSV('Data/' .. cancerType .. '/' .. pros[i+j-1])
			atoms = gene:size(1)
			for a=1,atoms do
				--print(gene[a][17] , gene[a][18])
				map[{ {}, {math.min(gene[a][17]+1,200)}, {math.min(gene[a][18]+1,200)} }] = gene[{ {a}, {1,16} }]
			end
			tens[j] = map:clone()
		end
		torch.save('Data/' .. cancerType .. '_Map/' .. pros[i]:sub(1,4) .. '.dat',tens)
	end
end

local function Shuffle(...)
	local args = {...}
	local n = #args[1]
	local count = 0
	count=(torch.type(n)=='number' and n or n[1]) 
	for t=1,count do
		local k = math.random(count)
		for i,v in ipairs{...} do
			v[t],v[k] = v[k],v[t]
		end
	end
	return {...}
end

function M.DataCollect(positive,negative,neutral)
	local posProtein = paths.dir('Data/' .. positive)
	table.remove(posProtein,1)
	table.remove(posProtein,1)
	local negProtein = paths.dir('Data/' .. negative)
	table.remove(negProtein,1)
	table.remove(negProtein,1)
	--local nutProtein = paths.dir('Data/' .. neutral)
	--table.remove(nutProtein,1)
	--table.remove(nutProtein,1)
	local label = torch.zeros(#posProtein+#negProtein)
	local allProtein = {}
	local index=1
	for i=1,#posProtein do
		table.insert(allProtein, 'Data/' .. positive .. '/' .. posProtein[i])
		label[index]=1
		index=index+1
	end

	for i=1,#negProtein do
		table.insert(allProtein, 'Data/' .. negative .. '/' .. negProtein[i])
	end

	--for i=1,#nutProtein do
	--	table.insert(allProtein, 'Data/' .. neutral .. '/' .. nutProtein[i])
	--end

	_ = Shuffle(allProtein,label)

	return allProtein,label
end

function M.Slice(tbl, first, last, step)
  local sliced = {}

  for i = first or 1, last or #tbl, step or 1 do
    sliced[#sliced+1] = tbl[i]
  end

  return sliced
end

function M.Fetch(names,h,w)
  local pros = torch.Tensor(3,#names,16,h,w)
  local threads = require 'threads'
  local pool = threads.Threads(4)
  for i=1,#names do
    pool:addjob(function()
      local temp = torch.load(names[i])
      return i,temp
    end,
    function(id,pr)
	--print(pr,pros:size())
      pros[{ {1}, {id}, {}, {}, {} }] = pr[1]:clone()
	  pros[{ {2}, {id}, {}, {}, {} }] = pr[2]:clone()
	  pros[{ {3}, {id}, {}, {}, {} }] = pr[3]:clone()
    end
    )
  end
  pool:synchronize()
  pool:terminate()
  return pros
end

function M.Dataset(proteins,labels,meani,stdvi)
	local dataset = {}
	dataset.data = proteins
	dataset.label = labels
	setmetatable(dataset,{__index = function(t, i) 
                    return {
                        t.data[i],
                        t.label[i]
                    } 
                end}
	);
	
	function dataset:size() 
	    return self.data:size(1) 
	end
	
	local mean = torch.zeros(16)
    local stdv  = torch.zeros(16)

	 if meani~=nil then
	   for i=1,16 do
       dataset.data:select(3, i):add(-meani[i])
       dataset.data:select(3, i):div(stdvi[i])
	   end
	 else	
		 for i=1,16 do
    	 mean[i] = dataset.data:select(3, i):mean() 
   		 dataset.data:select(3, i):add(-mean[i])
    	 stdv[i] = dataset.data:select(3, i):std()
    	 dataset.data:select(3, i):div(stdv[i])
		 end
	 end

	return dataset,mean,stdv
end

return M

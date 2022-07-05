
def test_init_ens_population_concept(n):
	p_add = 1- (1 / n)
	result = []
	for i in range(10000):
		result.append(1)
		while(random.random() < p_add):
			result[i] += 1
	return sum(result)/len(result)


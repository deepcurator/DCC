from rdflib import Graph, BNode, RDFS, RDF, URIRef, Literal
class GGenerator:
	pass

class OutputShapeGenerator(GGenerator):
	
	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/output_shape")
		self.attr_uriref2 = URIRef("http://example.org/output_shape2")
		self.output_shapes_list = []
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == '_output_shapes':
			
			if len(attr_list) == 8:
				if int(attr_list[3]) >= len(self.output_shapes_list):
					self.output_shapes_list.append([attr_list[7]])
				else:
					self.output_shapes_list[int(attr_list[3])].append(attr_list[7])
				# self.output_shapes.append(attr_list[7])	
				self.parsed = True
			
			elif len(attr_list) == 4:
				if int(attr_list[3]) >= len(self.output_shapes_list):
					self.output_shapes_list.append(['scalar'])
				else:
					self.output_shapes_list[int(attr_list[3])].append('scalar')
				# self.RDF_graph.add((sb, self.attr_uriref, shape))
				self.parsed = True

			return True
		return False

	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:

			for idx, output_shapes in enumerate(self.output_shapes_list):

				shape = Literal(",".join(output_shapes))
				if idx == 0:
					triples.append((sb, self.attr_uriref, shape))
				elif idx == 1:
					triples.append((sb, self.attr_uriref2, shape))

			self.clear()
		
		return triples

	def clear(self):
		self.parsed = False
		self.output_shapes_list = []

class ShapeGenerator(GGenerator):
	
	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/shape")
		self.attr_uriref2 = URIRef("http://example.org/shape2")
		self.shapes = []
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'shape':
			
			if len(attr_list) == 6:
				self.shapes.append([attr_list[5]])
				self.parsed = True
			
			elif len(attr_list) == 2:
				self.shapes.append(['scalar'])
				self.parsed = True
			
			return True
		return False

	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:

			for idx, shapes in enumerate(self.shapes):

				shape = Literal(",".join(shapes))
				if idx == 0:
					triples.append((sb, self.attr_uriref, shape))
				elif idx == 1:
					triples.append((sb, self.attr_uriref2, shape))

			self.clear()
		
		return triples

	def clear(self):
		self.parsed = False
		self.shapes = []

class TensorShapeGenerator(GGenerator):
	
	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/tensor_shape")
		self.attr_uriref2 = URIRef("http://example.org/tensor_shape2")
		self.tensor_shapes = []
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'value' and attr_list[1] == 'tensor' and attr_list[2] == 'tensorShape':
			
			if len(attr_list) == 7:
				self.tensor_shapes.append([attr_list[6]])
				self.parsed = True
			
			elif len(attr_list) == 3:
				self.tensor_shapes.append(['scalar'])
				self.parsed = True

			return True
		return False

	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:

			for idx, shapes in enumerate(self.tensor_shapes):

				shape = Literal(",".join(shapes))
				if idx == 0:
					triples.append((sb, self.attr_uriref, shape))
				elif idx == 1:
					triples.append((sb, self.attr_uriref2, shape))

			self.clear()
		
		return triples

	def clear(self):
		self.parsed = False
		self.tensor_shapes = []

class TTypeGenerator(GGenerator):
	
	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/T_type")
		self.T = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'T':
			self.T = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.T)))
			self.clear()
		
		return triples

	def clear(self):
		self.T = None
		self.parsed = False

class TransposeAGenerator(GGenerator):

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/transpose_a")
		self.transpose_a = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'transpose_a':
			self.transpose_a = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.transpose_a)))
			self.clear()
		
		return triples

	def clear(self):
		self.transpose_a = None
		self.parsed = False

class TransposeBGenerator(GGenerator):

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/transpose_b")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'transpose_b':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class DTypeGenerator(GGenerator):

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/dtype")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'dtype':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class TensorDTypeGenerator(GGenerator):

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/tensor_dtype")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'value' and attr_list[1] == 'tensor' and attr_list[2] == 'dtype':
			self.value = attr_list[3]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class TensorContentGenerator(GGenerator):

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/tensor_content")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'value' and attr_list[1] == 'tensor' and attr_list[2] == 'tensorContent':
			self.value = attr_list[3]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class TDimGenerator(GGenerator):

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/Tdim_dtype")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'Tdim':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class TidxGenerator(GGenerator):

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/Tidx")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'Tidx':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class TshapeGenerator(GGenerator):

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/Tshape_dtype")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'Tshape':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class NGenerator(GGenerator):

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/N")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'N':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class KeepDimsGenerator(GGenerator):

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/Keep_dims")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'keep_dims':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class ValidateIndicesGenerator(GGenerator):
	
	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/validate_indices")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'validate_indices':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class SummarizeGenerator(GGenerator):
	
	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/summarize")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'summarize':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class OutTypeGenerator(GGenerator):
	
	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/out_type")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'out_type':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class IndexTypeGenerator(GGenerator):
	
	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/index_type")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'index_type':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class DataFormatGenerator(GGenerator):
	
	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/data_format")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'data_format':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class UseLockingGenerator(GGenerator):

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/use_locking")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'use_locking':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class UseNesterovGenerator(GGenerator):

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/use_nesterov")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'use_nesterov':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class SharedNameGenerator(GGenerator):

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/shared_name")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'shared_name':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class ContainerGenerator(GGenerator):

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/container")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'container':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class SetOperationGenerator(GGenerator):

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/set_operation")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'set_operation':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class Seed2Generator(GGenerator):

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/seed2")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'seed2':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class SeedGenerator(GGenerator):

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/seed")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'seed':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class DstTGenerator(GGenerator):

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/DstT")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'DstT':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class SrcTGenerator(GGenerator):

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/SrcT")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'SrcT':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class TruncateGenerator(GGenerator):

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/Truncate")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'Truncate':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class OutputTypeGenerator(GGenerator): 

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/output_type")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'output_type':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class ValidateShapeGenerator(GGenerator): 

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/validate_shape")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'validate_shape':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class IndexGenerator(GGenerator): 

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/Index")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'Index':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class AxisGenerator(GGenerator): 

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/axis")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'axis':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class BeginMaskGenerator(GGenerator): 

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/begin_mask")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'begin_mask':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class EndMaskGenerator(GGenerator): 

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/end_mask")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'end_mask':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class NewAxisMaskGenerator(GGenerator): 

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/new_axis_mask")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'new_axis_mask':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class TmultiplesGenerator(GGenerator): 

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/Tmultiples")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'Tmultiples':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class TlabelsGenerator(GGenerator): 

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/Tlabels")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'Tlabels':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class ShrinkAxisMaskGenerator(GGenerator): 

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/shrink_axis_mask")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'shrink_axis_mask':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class MessageGenerator(GGenerator): 

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/message")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'message':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class EllipsisMaskGenerator(GGenerator): 

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/ellipsis_mask")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'ellipsis_mask':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class PaddingGenerator(GGenerator): 

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/padding")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'padding':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class TIGenerator(GGenerator): 

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/TI")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'TI':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class UseCUDNNOnGPUGenerator(GGenerator): 

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/use_cudnn_on_gpu")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'use_cudnn_on_gpu':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class DepthRadiusGenerator(GGenerator): 

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/depth_radius")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'depth_radius':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class AlphaGenerator(GGenerator): 

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/alpha")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'alpha':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class BetaGenerator(GGenerator): 

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/beta")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'beta':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class BiasGenerator(GGenerator): 

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/bias")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'bias':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class TPermGenerator(GGenerator): 

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/Tperm")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'Tperm':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class IdenticalElementShapesGenerator(GGenerator): 

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/identical_element_shapes")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'identical_element_shapes':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False

class ClearAfterReadGenerator(GGenerator): 

	def __init__(self):
		self.attr_uriref = URIRef("http://example.org/clear_after_read")
		self.value = None
		self.parsed = False

	def parse(self, attr_list):
		
		if attr_list[0] == 'clear_after_read':
			self.value = attr_list[2]
			self.parsed = True
			return True
		return False
			
	def get_results(self, sb):
		triples = [] 
		
		if self.parsed:
			triples.append((sb, self.attr_uriref, Literal(self.value)))
			self.clear()
		
		return triples

	def clear(self):
		self.value = None
		self.parsed = False


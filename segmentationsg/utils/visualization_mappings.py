import os,json


arxiv_classes = [
    'abstract',
    'affiliation',
    'author',
    'bibblock',
    'contentblock',
    'date',
    'document',
    'equation',
    'figure',
    'figurecaption',
    'figuregraphic',
    'foot',
    'head',
    'heading',
    'item',
    'itemize',
    'meta',
    'pagenr',
    'subject',
    'table',
    'tablecaption',
    'tabular',
]


#eperiodica_classes =  ['article', 'author', 'backgroundfigure', 'col', 'contentblock',
#                              'documentroot', 'figure', 'figurecaption', 'figuregraphic', 'foot',
#                              'footnote', 'head', 'header', 'introduction', 'item', 'itemize', 'logo',
#                              'meta', 'pagenr', 'row', 'table', 'tableofcontent', 'tabular', 'unk']

eperiodica_classes = ['article', 'author', 'backgroundfigure', 'col', 'contentblock', 'documentroot', 'figure', 'figurecaption', 'figuregraphic', 'foot', 'footnote', 'head', 'header', 'item', 'itemize', 'meta', 'orderedgroup', 'pagenr', 'row', 'table', 'tableofcontent', 'tabular', 'unorderedgroup']


all_classes = set(arxiv_classes + eperiodica_classes)

classes_not_in_arxiv = set(eperiodica_classes) - set(arxiv_classes)
classes_not_in_eperiodica = set(arxiv_classes) - set(eperiodica_classes)


print('all classes', all_classes)
print()
print('classes not in arxiv: ', classes_not_in_arxiv)
print()
print('classes not in eperiodica: ', classes_not_in_eperiodica)


class_names_remapped = {c: c for c in all_classes}
class_names_remapped['header'] = 'heading' #EP 'header' means a heading
class_names_remapped['head'] = 'header' #EP 'head' is the header
class_names_remapped['document'] = 'article'
class_names_remapped['unk'] = 'unknown'
class_names_remapped['col'] = 'column'
class_names_remapped['pagenr'] = 'page nr.'
class_names_remapped['backgroundfigure'] = 'background fig.'
class_names_remapped['figuregraphic'] = 'figure graphic'
class_names_remapped['figurecaption'] = 'figure caption'
class_names_remapped['tableofcontent'] = 'table of content'
class_names_remapped['tablecaption'] = 'table caption'
class_names_remapped['foot'] = 'footer'
class_names_remapped['contentblock'] = 'text block'
class_names_remapped['bibblock'] = 'bibliography block'
class_names_remapped['documentroot'] = 'document root'
class_names_remapped['subject'] = 'keywords'
class_names_remapped['orderedgroup'] = 'ordered group'
class_names_remapped['unorderedgroup'] = 'unordered group'


#remapped_class_name_list = sorted([class_names_remapped[c] for c in all_classes])
#reverse_class_names_remapped = {class_names_remapped[c]: c for c in all_classes}
#print('remapped_class_name_list', remapped_class_name_list)
#'abstract', 'affiliation', 'article', 'article', 'author', 'background fig.', 'bibliography block', 'column', 'date', 'document root', 'equation', 'figure', 'figure caption', 'figure graphic', 'footer', 'footnote', 'header', 'heading', 'heading', 'introduction', 'item', 'itemize', 'keywords', 'logo', 'meta', 'page nr.', 'row', 'table', 'table caption', 'table of content', 'tabular', 'text block', 'unknown']
import matplotlib.pyplot as plt
import matplotlib

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def rgb_to_hex(rgb):
    print(rgb)
    return '%02x%02x%02x' % rgb

#print(f'reverse_class_names_remapped: {reverse_class_names_remapped}')


#num_categories = len(remapped_class_name_list)
num_categories_max = len(all_classes)
remapped_class_names_list = list(class_names_remapped.values())
num_remapped_colors = len(remapped_class_names_list)
all_classes_list = list(all_classes)
random_color_map = get_cmap(num_remapped_colors)
color_mapping_dict = dict()
for i in range(num_categories_max):
    orig_class_name = all_classes_list[i]
    new_class_name = class_names_remapped[orig_class_name]
    color_index = remapped_class_names_list.index(new_class_name)
    class_color = random_color_map(color_index) 
    print(f"{i}; original name: {orig_class_name}; remapped name: {new_class_name}; RGB1: {class_color}; RGB2: {[int(x * 255) for x in class_color]}; HEX: { matplotlib.colors.to_hex(class_color)}")
    color_mapping_dict[orig_class_name] = class_color 

thing_classes_remapping_path = os.path.join(os.path.dirname(__file__), 'thing_classes_names_remapping_and_colors.json')
thing_classes_remapping_dict = {'class_names_to_new_names': class_names_remapped, 'class_names_to_colors': color_mapping_dict}

print('saving to', thing_classes_remapping_path)
with open(thing_classes_remapping_path, 'w') as f:
    json.dump(thing_classes_remapping_dict, f, indent=1)
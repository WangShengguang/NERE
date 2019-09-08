from py2neo import Graph, Node, Relationship, NodeMatcher


class ImportNeo4j:
    """Write the basic facts of the case on the structured text 
    and the triples obtained on the unstructured text to Neo4j"""

    def __init__(self):
        self.graph = Graph(
            host='172.27.10.140',
            http_port=7474,
            user='neo4j',
            password='szl'
        )
        # self.graph = Graph("bolt://172.27.10.140:7687", auth=("neo4j", "neo4j"))
        # self.graph.delete_all()  # clear database
        self.matcher = NodeMatcher(self.graph)
        self.case_id = 0

    def import_data(self, basic_fact, triples):
        entities = set()  # Guarantee the uniqueness of the entity node
        ent_labels = {}
        for triple in triples:
            e1_mention, e1_label = triple[0][0], triple[0][1]
            e2_mention, e2_label = triple[1][0], triple[1][1]
            if e1_mention not in ent_labels:
                ent_labels[e1_mention] = e1_label
            if e2_mention not in ent_labels:
                ent_labels[e2_mention] = e2_label
        self.write_basic_fact(basic_fact, entities, ent_labels)
        self.write_triples(triples, entities)

    def write_basic_fact(self, basic_fact, entities, ent_labels):
        if not basic_fact:
            return None

        self.case_id = basic_fact['编号']
        title = basic_fact['标题'] if '标题' in basic_fact else ''
        court = basic_fact['受理法院'] if '受理法院' in basic_fact else ''
        catergory = basic_fact['类型'] if '类型' in basic_fact else ''
        case_number = basic_fact['案号'] if '案号' in basic_fact else ''

        # Create a root node, a judgment usually has the attributes "title", "type" and "case number"
        node_root = Node('案件', name='案件', case_id=self.case_id, title=title, court=court,
                         catergory=catergory, case_number=case_number)
        self.graph.create(node_root)
        entities.add('案件')

        plaintiffs = basic_fact['原告']
        defendants = basic_fact['被告']
        self.write_litigant(plaintiffs, '原告', entities, ent_labels)
        self.write_litigant(defendants, '被告', entities, ent_labels)

    def write_litigant(self, litigants, relation, entities, ent_labels):
        """Write party information (plaint and defendant)
        Args:
            litigants: (list) party information
            relation: (str) "原告" or "被告"
        """
        for litigant in litigants:
            label = self.get_ent_label(litigant['名字'], ent_labels)
            node_litigant = Node(label, name=litigant['名字'], id=litigant['编号'], case_id=self.case_id)
            self.graph.create(node_litigant)
            entities.add(litigant['名字'])
            node_root = self.matcher.match('案件', name='案件', case_id=self.case_id).first()
            relationship = Relationship(node_root, relation, node_litigant, label='relation')
            self.graph.create(relationship)

            for item in litigant:
                if item != '编号' and item != '名字':
                    label = self.get_ent_label(litigant[item], ent_labels)
                    node_repr = Node(label, name=litigant[item], case_id=self.case_id)  # 代理人等
                    self.graph.create(node_repr)
                    entities.add(litigant[item])
                    relationship = Relationship(node_litigant, item, node_repr, label='relation')
                    self.graph.create(relationship)

    def get_ent_label(self, ent, ent_labels):
        if ent in ent_labels:
            label = ent_labels[ent]
        else:
            if len(ent) <= 4:
                label = '自然人主体'
            else:
                label = '非自然人主体'
        return label

    def write_triples(self, triples, entities):
        """Write the knowledge triples extracted from the unstructured text to neo4j"""
        for triple in triples:
            e1 = triple[0]
            e2 = triple[1]
            relation = triple[2]

            node1 = self.get_node(e1, entities)
            node2 = self.get_node(e2, entities)

            relationship = Relationship(node1, relation, node2, label='relation')
            self.graph.create(relationship)

    def get_node(self, ent, entities):
        """Get a unique entity node, create or get if it already exists"""
        ent_mention, ent_label = ent[0], ent[1]
        if ent_mention not in entities:
            node = Node(ent_label, name=ent_mention, case_id=self.case_id)
            self.graph.create(node)
            entities.add(ent_mention)
        else:
            node = self.matcher.match(ent_label, name=ent_mention, case_id=self.case_id).first()
        return node

import logging

from argparse import ArgumentParser

from ..core.labels import MultiLabeler

logger = logging.getLogger('nf.test')


def add_args(module_name: str, parser: ArgumentParser) -> None:
    pass


def test_init(self):
    labeler: MultiLabeler = MultiLabeler(labels=['eco', 'lab', 'wel', 'sec', 'cul'])
    self.assertEqual(['',
                      'eco',
                      'lab',
                      'ecolab',
                      'wel',
                      'ecowel',
                      'labwel',
                      'ecolabwel',
                      'sec',
                      'ecosec',
                      'labsec',
                      'ecolabsec',
                      'welsec',
                      'ecowelsec',
                      'labwelsec',
                      'ecolabwelsec',
                      'cul',
                      'ecocul',
                      'labcul',
                      'ecolabcul',
                      'welcul',
                      'ecowelcul',
                      'labwelcul',
                      'ecolabwelcul',
                      'seccul',
                      'ecoseccul',
                      'labseccul',
                      'ecolabseccul',
                      'welseccul',
                      'ecowelseccul',
                      'labwelseccul',
                      'ecolabwelseccul'], labeler.labels)


def test_decode(self):
    labeler: MultiLabeler = MultiLabeler(labels=['eco', 'lab', 'wel', 'sec', 'cul'])
    labels = labeler.decode(18)
    self.assertEqual(['lab', 'cul'], labels)

    labels = labeler.decode(19)
    self.assertEqual(['eco', 'lab', 'cul'], labels)

    labels = labeler.decode(20)
    self.assertEqual(['wel', 'cul'], labels)


def test_binpowset(self):
    labeler: MultiLabeler = MultiLabeler(labels=['eco', 'lab', 'wel', 'sec', 'cul'])
    labels = labeler.binpowset(18)
    self.assertEqual([0, 1, 0, 0, 1], labels)

    labels = labeler.binpowset(19)
    self.assertEqual([1, 1, 0, 0, 1], labels)

    labels = labeler.binpowset(20)
    self.assertEqual([0, 0, 1, 0, 1], labels)


def test_encode(self):
    labeler: MultiLabeler = MultiLabeler(labels=['eco', 'lab', 'wel', 'sec', 'cul'])
    idx = labeler.encode(['eco', 'lab'])
    self.assertEqual(3, idx)

    idx = labeler.encode(['lab', 'eco'])
    self.assertEqual(3, idx)

    idx = labeler.encode(['cul', 'sec', 'lab'])
    self.assertEqual(26, idx)


def test_sample(args) -> int:
    pass
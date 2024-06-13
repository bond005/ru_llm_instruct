import os
import sys
import unittest

try:
    from wiki_parsing.wiki_parsing import load_wiki_file, remove_wiki_links
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from wiki_parsing.wiki_parsing import load_wiki_file, remove_wiki_links


class TestWikiParsing(unittest.TestCase):
    def test_load_wiki_file(self):
        wiki_fname = os.path.join(os.path.dirname(__file__), 'testdata', 'wiki_sample.txt')
        true_titles = ['Социология', 'Киевская Русь', 'Путь из варяг в греки']
        loaded = load_wiki_file(wiki_fname)
        print(f'\n{loaded}')  # for debug
        set_of_paragraphs = set()
        set_of_sections = set()
        self.assertIsInstance(loaded, list)
        self.assertEqual(len(loaded), len(true_titles))
        for idx, val in enumerate(loaded):
            self.assertIsInstance(val, dict)
            self.assertEqual(set(val.keys()), {'title', 'document'})
            self.assertIsInstance(val['title'], str)
            self.assertEqual(true_titles[idx], val['title'])
            self.assertIsInstance(val['document'], list)
            self.assertGreater(len(val['document']), 0)
            for section in val['document']:
                self.assertIsInstance(section, dict)
                self.assertEqual(set(section.keys()), {'section_title', 'section_body'})
                self.assertIsInstance(section['section_title'], str)
                self.assertIsInstance(section['section_body'], list)
                self.assertGreater(len(section['section_title'].strip()), 0)
                self.assertGreater(len(section['section_body']), 0)
                for paragraph in section['section_body']:
                    self.assertIsInstance(paragraph, str)
                    self.assertGreater(len(paragraph.strip()), 0)
                    self.assertNotIn(paragraph, set_of_paragraphs)
                    set_of_paragraphs.add(paragraph)
                self.assertNotIn(section['section_title'], set_of_sections)
                set_of_sections.add(section['section_title'])

    def test_remove_wiki_links_pos01(self):
        source_text = ('По сведениям [[Константин Багрянородный|Константина Багрянородного]] (X век), '
                       '[[кривичи]] и другие племена весной возили в Милиниску ([[Смоленск]]) и '
                       'Чернигогу ([[Чернигов]]) большие долблёные ладьи на '
                       '30-40 человек — [[Долблёнка|однодерёвки]], которые затем сплавлялись по Днепру в [[Киев]].')
        target_text = ('По сведениям Константина Багрянородного (X век), кривичи и другие племена весной возили в '
                       'Милиниску (Смоленск) и Чернигогу (Чернигов) большие долблёные ладьи на 30-40 человек — '
                       'однодерёвки, которые затем сплавлялись по Днепру в Киев.')
        self.assertEqual(remove_wiki_links(source_text), target_text)

    def test_remove_wiki_links_pos02(self):
        source_text = ('По сведениям Константина Багрянородного (X век), кривичи и другие племена весной возили в '
                       'Милиниску (Смоленск)  и Чернигогу (Чернигов) большие долблёные ладьи на 30-40 человек — '
                       'однодерёвки, которые затем сплавлялись по Днепру в  Киев.')
        target_text = ('По сведениям Константина Багрянородного (X век), кривичи и другие племена весной возили в '
                       'Милиниску (Смоленск) и Чернигогу (Чернигов) большие долблёные ладьи на 30-40 человек — '
                       'однодерёвки, которые затем сплавлялись по Днепру в Киев.')
        self.assertEqual(remove_wiki_links(source_text), target_text)

    def test_remove_wiki_links_neg01(self):
        self.assertEqual(remove_wiki_links(''), '')




if __name__ == '__main__':
    unittest.main(verbosity=2)

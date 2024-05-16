import unittest
from chain import rag_chain

class TestRAGChain(unittest.TestCase):

    def test_rag_chain(self):
        ans = rag_chain.invoke("Who are Meta's 'Directors' (i.e., members of the Board of Directors)?")
        self.assertTrue(ans != "Peggy Alford, Marc L. Andreessen, Andrew W. Houston, Nancy Killefer, Robert M. Kimmitt, Sheryl K. Sandberg, Tracey T. Travis, Tony Xu")
                        
if __name__ == '__main__':
    unittest.main()
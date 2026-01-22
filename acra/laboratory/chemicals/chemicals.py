import pubchempy as pcp
from selenium import webdriver
from selenium.webdriver.common.by import By


class Chemical:
    def __init__(self, name, approx_search=True):
        self.approx_search = approx_search
        self.name = name  # name of the chemical form procedure
        self.pubchem_template = "https://pubchem.ncbi.nlm.nih.gov/#query="
        self.cid = None
        self.smiles = None
        self.inchi = None
        self.inchikey = None
        self.iupac_name = None
        self.molecular_formula = None
        self.molecular_weight = None
        self.driver = None
        self.compound = None

    def _init_driver(self):
        options = webdriver.FirefoxOptions()
        options.add_argument("--headless")
        self.driver = webdriver.Firefox(options=options)

    def _unkown_to_cid(self):
        # request to pubchem
        if self.driver is None:
            self._init_driver()
        url = self.pubchem_template + f'"{self.name}"'
        self.driver.get(url)
        try:
            href = self.driver.find_element(
                By.XPATH,
                '//a[starts-with(@href, "https://pubchem.ncbi.nlm.nih.gov/compound/") and number(substring-after(@href, "https://pubchem.ncbi.nlm.nih.gov/compound/"))]',
            )
            cid = href.get_attribute('href').split("/")[-1]
        except Exception as _:
            cid = None
            return False

        if cid:
            self.cid = int(cid)
            return True
        else:
            self.cid = None
            return False

    def get_cid(self):
        c = pcp.get_compounds(self.name, "name")
        if len(c) > 0:
            self.cid = c[0].cid
            return True
        elif self.approx_search:
            found = self._unkown_to_cid()
            if not found:
                self.data = None
                return False
        else:
            self.cid = None
            return False
        return False

    def get_molecule_properties(self):
        self.compound = pcp.Compound.from_cid(self.cid)
        self.molecular_weight = self.compound.molecular_weight  # [g/mol]
        self.iupac_name = self.compound.iupac_name
        return None

    def get_pubchem_data(self):
        if self.cid is not None:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{self.cid}/JSON"
            data = pcp._request(url)
            self.data = data
        return None

    def mol_to_mass(self, mol: float):
        """
        mol: float
            moles of the compound in mol
        """
        return mol * self.molecular_weight
    
    def cid_to_abstract(self):
        """
        Get abstract from pubchem
        """
        abstract = pcp.get_synonyms(self.cid)
        return abstract
        

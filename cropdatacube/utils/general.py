import glob
import os

from typing import Tuple, List, Union
import random
from dateutil.parser import parse

def find_postinlist(listvalues: List[float], refvalue: float) -> Union[int, None]:
    """
    Find the position of a reference value in a sorted list.

    Parameters
    ----------
    listvalues : list of float
        List of values in which the search will be performed. This list will be sorted in-place.
    refvalue : float
        The value to search for in the list.

    Returns
    -------
    int or None
        The position of the reference value in the list if found, otherwise None.

    Notes
    -----
    This function assumes that the list values are numeric and sorts the list in ascending order.
    """
    listvalues.sort()
    posinlist = [i for i in range(len(listvalues)-1) if refvalue <= listvalues[i+1] and refvalue >= listvalues[i]]
    if len(posinlist)==0:
        posinlist = None
    else:
        posinlist = posinlist[0]
    return posinlist


def list_files(input_path: str, pattern: str = "xml") -> List[str]:
    """
    List all files in a directory and its subdirectories that match a given pattern.

    Parameters
    ----------
    input_path : str
        The path to the directory where the search will be performed.
    pattern : str, optional
        The pattern to match the filenames. Defaults to 'xml'.

    Returns
    -------
    List[str]
        A list of file paths that match the given pattern.
    
    Notes
    -----
    This function uses the `glob` module to perform recursive searches within the given directory.
    """
    files = glob.glob(os.path.join(input_path, '**', f'*{pattern}'), recursive=True)
    return [f for f in files if os.path.isfile(f)]


def split_filename(filename) -> Tuple[str, str]:
    """fal to know if the given filenmae is it a filename or it also includes the path

    Args:
        filename (str): file name

    Returns:
        tmppath: if the given name includes the directory path then it will save the path in a different variable
        fn: filename
        
    """
    dirname = os.path.dirname(filename)
    if dirname != '':
        tmppath = dirname
        fn = os.path.basename(filename)
    else:
        tmppath = None
        
    return tmppath, fn




def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try: 
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False



def find_date_instring(string, pattern = "202", yearformat = 'yyyy'):
    """find date pattern in a string

    Args:
        string (_type_): string
        pattern (str, optional): date init. Defaults to "202".
        yearformat (str, optional): the year format in the string 2021 is yyyy. Defaults to 'yyyy'.

    Returns:
        string: date in yyyymmdd format
    """
    matches = re.finditer(pattern, string)
    
    if yearformat == 'yyyy':
        datelen = 8
    else:
        datelen = 6
    
    matches_positions = [string[match.start():match.start() +datelen] 
                                for match in matches if is_date(string[match.start():match.start() +datelen])]
    if len(matches_positions[0]) == 6:
        matches_positions = ['20'+matches_positions[0]]
    
    return matches_positions[0]

class FolderWithImages(object):
    """
    A class to manage a folder containing image files.

    Attributes
    ----------
    path : str
        The path to the folder containing image files.
    imgs_suffix : str
        The suffix of image files to look for (e.g., '.jpg').
    shuffle : bool
        Whether to shuffle the list of image files.
    seed : int, optional
        The seed for the random number generator used in shuffling.

    Methods
    -------
    length
        Returns the number of image files in the folder.
    files_in_folder
        Returns a list of image files in the folder.
    _look_for_images()
        Looks for image files in the folder and optionally shuffles them.
    """
    
    @property
    def length(self):
        """
        Returns the number of image files in the folder.

        Returns
        -------
        int
            The number of image files in the folder.
        """
        return len(self._look_for_images())

    @property
    def files_in_folder(self):
        """
        Returns a list of image files in the folder.

        Returns
        -------
        list of str
            A list of paths to image files in the folder.
        """
        return self._look_for_images()
        
    def _look_for_images(self):
        """
        Looks for image files in the folder and optionally shuffles them.

        Returns
        -------
        list of str
            A list of paths to image files in the folder.

        Raises
        ------
        ValueError
            If no image files are found in the folder.
        """
        filesinfolder = list_files(self.path, pattern= self.imgs_suffix)
        if len(filesinfolder)==0:
            raise ValueError(f'there are not images in this {self.path} folder')

        if self.seed is not None and self.shuffle:
            random.seed(self.seed)
            random.shuffle(filesinfolder)

        elif self.shuffle:
            random.shuffle(filesinfolder)

        return filesinfolder

    def __init__(self, path,suffix = '.jpg', shuffle = False, seed = None) -> None:
        """
        Initialize the FolderWithImages object.

        Parameters
        ----------
        path : str
            The path to the folder containing image files.
        suffix : str, optional
            The suffix of image files to look for (default is '.jpg').
        shuffle : bool, optional
            Whether to shuffle the list of image files (default is False).
        seed : int, optional
            The seed for the random number generator used in shuffling (default is None).
        """
        self.path = path
        self.imgs_suffix = suffix
        self.shuffle = shuffle
        self.seed = seed

from typing import Any

from pandas import DataFrame

class Notebook:
    def __init__(self) -> None:
        self.data = []

    def write(self, input_data: Any, short_description: str):
        self.data.append({"Short Description": short_description, "Content": self._normalize_content(input_data)})
        return f"The information has been recorded in Notebook, and its index is {len(self.data)-1}."
    
    def update(self, input_data: Any, index: int, short_decription: str):
        self.data[index]["Content"] = self._normalize_content(input_data)
        self.data[index]["Short Description"]  = short_decription

        return f"The information has been updated in Notebook."
    
    def list(self):
        results = []
        for idx, unit in enumerate(self.data):
            results.append({"index":idx, "Short Description":unit['Short Description']})
        
        return results

    def list_all(self):
        results = []
        for idx, unit in enumerate(self.data):
            results.append({
                "index": idx,
                "Short Description": unit["Short Description"],
                "Content": self._normalize_content(unit["Content"]),
            })
        
        return results
    
    def read(self, index):
        return self.data[index]
    
    def reset(self):
        self.data = []

    def _normalize_content(self, input_data):
        if isinstance(input_data, DataFrame):
            return input_data.to_dict(orient="records")
        return input_data
    
    

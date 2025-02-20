import ipaddress
import re
import urllib.request
from bs4 import BeautifulSoup
import socket
import requests
from googlesearch import search
import whois
from datetime import date, datetime
from urllib.parse import urlparse

class FeatureExtraction:
    features = []
    
    def __init__(self, url):
        self.features = []
        self.url = url
        self.domain = ""
        self.whois_response = ""
        self.urlparse = ""
        self.response = ""
        self.soup = ""

        try:
            self.response = requests.get(url)
            self.soup = BeautifulSoup(self.response.text, 'html.parser')  # Corrected to use self.response
        except:
            pass

        try:
            self.urlparse = urlparse(url)
            self.domain = self.urlparse.netloc
        except:
            pass

        try:
            self.whois_response = whois.whois(self.domain)
        except:
            pass

        # Append feature extraction methods to the feature list
        self.features.append(self.UsingIp())
        self.features.append(self.longUrl())
        self.features.append(self.shortUrl())
        self.features.append(self.symbol())
        self.features.append(self.redirecting())
        self.features.append(self.prefixSuffix())
        self.features.append(self.SubDomains())
        self.features.append(self.Hppts())
        self.features.append(self.DomainRegLen())
        self.features.append(self.Favicon())
        self.features.append(self.NonStdPort())
        self.features.append(self.HTTPSDomainURL())
        self.features.append(self.RequestURL())
        self.features.append(self.AnchorURL())
        self.features.append(self.LinksInScriptTags())
        self.features.append(self.ServerFormHandler())
        self.features.append(self.InfoEmail())
        self.features.append(self.AbnormalURL())
        self.features.append(self.WebsiteForwarding())
        self.features.append(self.StatusBarCust())
        self.features.append(self.DisableRightClick())
        self.features.append(self.UsingPopupWindow())
        self.features.append(self.IframeRedirection())
        self.features.append(self.AgeofDomain())
        self.features.append(self.DNSRecording())
        self.features.append(self.WebsiteTraffic())
        self.features.append(self.PageRank())
        self.features.append(self.GoogleIndex())
        self.features.append(self.LinksPointingToPage())
        self.features.append(self.StatsReport())

    # 1. UsingIp
    def UsingIp(self):
        try:
            ipaddress.ip_address(self.url)
            return -1
        except:
            return 1

    # 2. longUrl
    def longUrl(self):
        if len(self.url) < 54:
            return 1
        if len(self.url) >= 54 and len(self.url) <= 75:
            return 0
        return -1

    # 3. shortUrl
    def shortUrl(self):
        match = re.search(r'bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                    r'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                    r'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                    r'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                    r'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                    r'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                    r'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|tr\.im|link\.zip\.net', self.url)
        if match:
            return -1
        return 1

    # 4. Symbol@
    def symbol(self):
        if re.findall("@", self.url):
            return -1
        return 1
    
    # 5. Redirecting//
    def redirecting(self):
        if self.url.rfind('//') > 6:
            return -1
        return 1
    
    # 6. prefixSuffix
    def prefixSuffix(self):
        try:
            match = re.findall('-', self.domain)
            if match:
                return -1
            return 1
        except:
            return -1
    
    # 7. SubDomains
    def SubDomains(self):
        dot_count = len(re.findall(r"\.", self.url))
        if dot_count == 1:
            return 1
        elif dot_count == 2:
            return 0
        return -1

    # 8. HTTPS
    def Hppts(self):
        try:
            https = self.urlparse.scheme
            if 'https' in https:
                return 1
            return -1
        except:
            return 1

    # 9. DomainRegLen
    def DomainRegLen(self):
        try:
            expiration_date = self.whois_response.expiration_date
            creation_date = self.whois_response.creation_date
            try:
                if len(expiration_date):
                    expiration_date = expiration_date[0]
            except:
                pass
            try:
                if len(creation_date):
                    creation_date = creation_date[0]
            except:
                pass

            age = (expiration_date.year - creation_date.year) * 12 + (expiration_date.month - creation_date.month)
            if age >= 12:
                return 1
            return -1
        except:
            return -1

    # 10. Favicon
    def Favicon(self):
        try:
            for head in self.soup.find_all('head'):
                for link in self.soup.find_all('link', href=True):
                    dots = [x.start(0) for x in re.finditer(r'\.', link['href'])]
                    if self.url in link['href'] or len(dots) == 1 or self.domain in link['href']:
                        return 1
            return -1
        except:
            return -1

    # 11. NonStdPort
    def NonStdPort(self):
        try:
            port = self.domain.split(":")
            if len(port) > 1:
                return -1
            return 1
        except:
            return -1

    # 12. HTTPSDomainURL
    def HTTPSDomainURL(self):
        try:
            if 'https' in self.domain:
                return -1
            return 1
        except:
            return -1
    
    # 13. RequestURL
    def RequestURL(self):
        try:
            success, i = 0, 0
            for img in self.soup.find_all('img', src=True):
                dots = [x.start(0) for x in re.finditer(r'\.', img['src'])]
                if self.url in img['src'] or self.domain in img['src'] or len(dots) == 1:
                    success += 1
                i += 1

            for audio in self.soup.find_all('audio', src=True):
                dots = [x.start(0) for x in re.finditer(r'\.', audio['src'])]
                if self.url in audio['src'] or self.domain in audio['src'] or len(dots) == 1:
                    success += 1
                i += 1

            for embed in self.soup.find_all('embed', src=True):
                dots = [x.start(0) for x in re.finditer(r'\.', embed['src'])]
                if self.url in embed['src'] or self.domain in embed['src'] or len(dots) == 1:
                    success += 1
                i += 1

            for iframe in self.soup.find_all('iframe', src=True):
                dots = [x.start(0) for x in re.finditer(r'\.', iframe['src'])]
                if self.url in iframe['src'] or self.domain in iframe['src'] or len(dots) == 1:
                    success += 1
                i += 1

            try:
                percentage = success / float(i) * 100
                if percentage < 22.0:
                    return 1
                elif 22.0 <= percentage < 61.0:
                    return 0
                else:
                    return -1
            except:
                return -1
        except:
            return -1

    # 14. AnchorURL
    def AnchorURL(self):
        try:
            anchor = 0
            for anchor_tag in self.soup.find_all('a', href=True):
                if self.url in anchor_tag['href'] or self.domain in anchor_tag['href']:
                    anchor += 1
            return -1 if anchor > 1 else 1
        except:
            return -1

    # 15. LinksInScriptTags
    def LinksInScriptTags(self):
        try:
            script = 0
            for script_tag in self.soup.find_all('script', src=True):
                if self.url in script_tag['src'] or self.domain in script_tag['src']:
                    script += 1
            return -1 if script > 0 else 1
        except:
            return 1

    # 16. ServerFormHandler
    def ServerFormHandler(self):
        try:
            action = self.soup.find_all('form', action=True)
            for action_tag in action:
                if self.url in action_tag['action']:
                    return -1
            return 1
        except:
            return 1

    # 17. InfoEmail
    def InfoEmail(self):
        try:
            emails = re.findall(r"@[\w-]+\.[\w-]+", self.url)
            if emails:
                return -1
            return 1
        except:
            return 1

    # 18. AbnormalURL
    def AbnormalURL(self):
        try:
            if re.findall(r'[^A-Za-z0-9]+', self.url):
                return -1
            return 1
        except:
            return 1

    # 19. WebsiteForwarding
    def WebsiteForwarding(self):
        try:
            if self.url == self.domain:
                return 1
            return -1
        except:
            return -1

    # 20. StatusBarCust
    def StatusBarCust(self):
        try:
            status_bar = self.soup.find_all('script', type='text/javascript')
            if status_bar:
                return -1
            return 1
        except:
            return 1

    # 21. DisableRightClick
    def DisableRightClick(self):
        try:
            scripts = self.soup.find_all('script')
            for script_tag in scripts:
                if re.findall('oncontextmenu', script_tag.text):
                    return -1
            return 1
        except:
            return 1

    # 22. UsingPopupWindow
    def UsingPopupWindow(self):
        try:
            if re.findall(r"window\.open", self.soup.text):
                return -1
            return 1
        except:
            return 1

    # 23. IframeRedirection
    def IframeRedirection(self):
        try:
            iframe = self.soup.find_all('iframe', src=True)
            for iframe_tag in iframe:
                if self.url in iframe_tag['src'] or self.domain in iframe_tag['src']:
                    return -1
            return 1
        except:
            return 1

    # 24. AgeofDomain
    def AgeofDomain(self):
        try:
            creation_date = self.whois_response.creation_date
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
            age = datetime.now() - creation_date
            if age.days > 365:
                return 1
            return -1
        except:
            return -1

    # 25. DNSRecording
    def DNSRecording(self):
        try:
            if self.whois_response and self.whois_response.domain_name:
                return 1
            return -1
        except:
            return -1

    # 26. WebsiteTraffic
    def WebsiteTraffic(self):
        try:
            rank = self.PageRank()
            if rank == -1:
                return -1
            return 1
        except:
            return -1

    # 27. PageRank
    def PageRank(self):
        try:
            rank_checker_response = requests.get(f"https://www.google.com/search?q=site:{self.url}")
            if "PageRank" in rank_checker_response.text:
                return 1
            return -1
        except:
            return -1

    # 28. GoogleIndex
    def GoogleIndex(self):
        try:
            for result in search(self.url, num_results=1):
                if self.url in result:
                    return 1
            return -1
        except:
            return -1

    # 29. LinksPointingToPage
    def LinksPointingToPage(self):
        try:
            if len(self.soup.find_all('a')) > 0:
                return 1
            return -1
        except:
            return -1

    # 30. StatsReport
    def StatsReport(self):
        try:
            stats = re.findall(r"\d{2,3}\.\d{2,3}\.\d{2,3}\.\d{2,3}", self.url)
            if stats:
                return -1
            return 1
        except:
            return 1

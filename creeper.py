import argparse, os, hashlib, re
from pprint import pprint

import requests, bs4
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--number', '-n', default=10)
    parser.add_argument('--data_path', default='data')

    args = parser.parse_args()

    abs_path = os.path.join(args.data_path, 'abs')
    pdf_path = os.path.join(args.data_path, 'pdf')

    # make directories
    if not os.path.exists(args.data_path): os.makedirs(args.data_path)
    if not os.path.exists(abs_path): os.makedirs(abs_path)
    if not os.path.exists(pdf_path): os.makedirs(pdf_path)

    site = 'https://arxiv.org/list/cs/pastweek?show=' + str(args.number)

    print('connect to ' + site)

    def get_website(url):
        r = requests.get(url)

        if r.status_code == 200:
            return r.text
        else:
            raise Exception('Invalid url: ' + url)

    text = get_website(site)
    bs = BeautifulSoup(text, 'html5lib')

    dts = bs('dt') # paper serial number
    dds = bs('dd') # paper metadata

    cnt = 0

    with open(os.path.join(args.ata_path, 'papar_list.txt'), 'w') as fp:

        for dt, dd in zip(dts, dds):

            if cnt >= args.number:
                break

            # save paper serial number and title
            sno = dt.span.a.string
            title = dd.div.div.contents[-1].split()
            st = ' '.join([sno] + title)

            h = hashlib.sha1()
            h.update(st.encode())
            hs = h.hexdigest()

            # download abstract text
            url = 'https://arxiv.org' + dt.span('a')[0]['href']
            abs_bs = BeautifulSoup(get_website(url), 'html5lib')
            abs_text = ''

            for t in abs_bs.blockquote.contents[2:]:

                if type(t) is bs4.element.Tag:

                    if t.string is not None:
                        abs_text += t.string
                else:
                    abs_text += t

            abs_text = ' '.join(abs_text.split())
            abs_text = '\n'.join(sent_tokenize(abs_text))

            with open(os.path.join(abs_path, hs + '.abs'), 'w') as fp_:
                fp_.writelines(abs_text)

            # download pdf file
            url = 'https://arxiv.org' + dt.span('a')[1]['href']
            r = requests.get(url)

            if r.status_code == 200:

                with open(os.path.join(pdf_path, hs + '.pdf'), 'wb') as fp_:
                    fp_.write(r.content)

                fp.write(st + '\n')
            else:
                raise Exception('PDF download failed: %s' % (url))

            print(st)

            cnt += 1

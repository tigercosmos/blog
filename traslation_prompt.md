I will provide a directory of posts, translate all posts under the directory.

directory: `source/_posts/c++`

## Steps
1. list the posts in the directory. `scripts/generate-translation-list.js` will help you to get the latest status.
2. translate the post into english (EN) and japanese (JP).
3. once finish a translation, check the translation checkbox, such as EN or JP.
4. once finish all translation of a post, ensure all checkboxes are checked, and check the item (post).

## Rules of translation

1. Modify the original post

the original post need to insert 
```
lang: <original_lang>
translation_key: <unique_key>
```

- language is usually `ZH`
- always use the file name as the unique key, for example "abc_123.md" has `abc_123` as the key.

2. Add a translation for EN and JP

- Write in **professional and technical style** and meanwhile **keep the original tone**.
- The title always needs to use quote, for example: `title: "some translation title"`. This is to prevent parsing error of title string.
- Translation needs to **keep all the original text and images**. 
- **Never try to shorten or summarize the content.** Polishing the sentence is fine.
- Be careful when handling images or HTML.

3. Translations will be put under `source/_posts_translation/<lang>/<original_path>`
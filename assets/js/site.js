// Small behaviors shared by every page: external links, heading anchors, copy buttons on code,
// footnotes on hover, the post's contents and reading progress, and the tag filter on the blog index.
(function () {
  "use strict";

  const prose = document.querySelector(".prose");
  const isPost = document.body.classList.contains("post");

  // Links to other sites open in a new tab
  document.querySelectorAll("a[href^='http']").forEach((a) => {
    if (a.hostname !== location.hostname) {
      a.target = "_blank";
      a.rel = "noopener";
    }
  });

  // A "#" beside each heading, to link to that section
  if (prose) {
    prose.querySelectorAll("h1[id], h2[id], h3[id]").forEach((h) => {
      const a = document.createElement("a");
      a.className = "anchor";
      a.href = "#" + h.id;
      a.textContent = "#";
      a.setAttribute("aria-label", "Link to this section");
      h.appendChild(a);
    });
  }

  // A copy button on every code block
  document.querySelectorAll("div.highlighter-rouge").forEach((block) => {
    const code = block.querySelector("pre");
    if (!code || !navigator.clipboard) return;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "copy-btn";
    btn.textContent = "Copy";
    btn.addEventListener("click", () => {
      navigator.clipboard.writeText(code.innerText).then(() => {
        btn.textContent = "Copied";
        setTimeout(() => (btn.textContent = "Copy"), 1500);
      });
    });
    block.appendChild(btn);
  });

  // Footnotes: hovering (or focusing) the number shows the note in a small card beside it
  let card = null;
  let hideTimer = null;
  const hideCard = () => {
    if (card) card.remove();
    card = null;
  };
  document.querySelectorAll("a.footnote").forEach((ref) => {
    const note = document.getElementById(decodeURIComponent(ref.hash.slice(1)));
    if (!note) return;
    const show = () => {
      clearTimeout(hideTimer);
      hideCard();
      card = document.createElement("div");
      card.className = "fn-card";
      card.innerHTML = note.innerHTML;
      card.querySelectorAll(".reversefootnote").forEach((el) => el.remove());
      card.addEventListener("mouseenter", () => clearTimeout(hideTimer));
      card.addEventListener("mouseleave", hideCard);
      document.body.appendChild(card);
      const r = ref.getBoundingClientRect();
      const left = Math.max(16, Math.min(r.left + window.scrollX - 20, window.scrollX + window.innerWidth - card.offsetWidth - 16));
      card.style.left = left + "px";
      card.style.top = r.bottom + window.scrollY + 8 + "px";
    };
    const hideSoon = () => (hideTimer = setTimeout(hideCard, 200));
    ref.addEventListener("mouseenter", show);
    ref.addEventListener("focus", show);
    ref.addEventListener("mouseleave", hideSoon);
    ref.addEventListener("blur", hideSoon);
  });

  if (isPost && prose) {
    // The contents, built from the post's h2 and h3, with the section being read marked
    const toc = document.querySelector(".toc");
    const heads = Array.from(prose.querySelectorAll("h2[id], h3[id]")).filter((h) => !h.closest(".footnotes"));
    const links = new Map();
    if (toc && heads.length >= 2) {
      heads.forEach((h) => {
        const a = document.createElement("a");
        a.href = "#" + h.id;
        a.textContent = h.firstChild.textContent;
        if (h.tagName === "H3") a.className = "toc--h3";
        toc.appendChild(a);
        links.set(h, a);
      });
      toc.hidden = false;
    }

    const bar = document.querySelector(".progress__bar");
    let ticking = false;
    const update = () => {
      ticking = false;
      const top = prose.offsetTop;
      const span = prose.offsetHeight - window.innerHeight;
      const done = span > 0 ? Math.min(1, Math.max(0, (window.scrollY - top) / span)) : 0;
      if (bar) bar.style.transform = "scaleX(" + done + ")";
      if (toc) toc.classList.toggle("is-visible", window.scrollY > top - 200);
      let current = null;
      heads.forEach((h) => {
        if (h.getBoundingClientRect().top < 120) current = h;
      });
      links.forEach((a, h) => a.classList.toggle("is-active", h === current));
    };
    window.addEventListener("scroll", () => {
      if (!ticking) {
        ticking = true;
        requestAnimationFrame(update);
      }
    }, { passive: true });
    window.addEventListener("resize", update);
    update();
  }

  // Blog index: a tag shows only the posts that carry it
  const filter = document.querySelector(".tag-filter");
  if (filter) {
    filter.addEventListener("click", (e) => {
      const btn = e.target.closest(".tag");
      if (!btn) return;
      filter.querySelectorAll(".tag").forEach((b) => b.classList.toggle("is-active", b === btn));
      const tag = btn.dataset.tag;
      document.querySelectorAll(".post-list__item").forEach((li) => {
        li.hidden = tag !== "" && !li.dataset.tags.split("|").includes(tag);
      });
    });
  }
})();
